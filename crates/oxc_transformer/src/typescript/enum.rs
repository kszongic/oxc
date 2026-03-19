use std::cell::Cell;

use oxc_allocator::{TakeIn, Vec as ArenaVec};
use oxc_ast::{NONE, ast::*};
use oxc_ast_visit::{VisitMut, walk_mut};
use oxc_data_structures::stack::NonEmptyStack;
use oxc_semantic::{ScopeFlags, ScopeId};
use oxc_span::{Ident, SPAN, Span};
use oxc_syntax::{
    constant_value::ConstantValue,
    number::NumberBase,
    operator::{AssignmentOperator, BinaryOperator, LogicalOperator, UnaryOperator},
    reference::ReferenceFlags,
    symbol::SymbolFlags,
};
use oxc_traverse::{BoundIdentifier, Traverse};

use crate::{context::TraverseCtx, state::TransformState};

pub struct TypeScriptEnum {
    optimize_const_enums: bool,
}

impl TypeScriptEnum {
    pub fn new(optimize_const_enums: bool) -> Self {
        Self { optimize_const_enums }
    }
}

impl<'a> Traverse<'a, TransformState<'a>> for TypeScriptEnum {
    fn enter_statement(&mut self, stmt: &mut Statement<'a>, ctx: &mut TraverseCtx<'a>) {
        let new_stmt = match stmt {
            Statement::TSEnumDeclaration(ts_enum_decl) => {
                self.transform_ts_enum(ts_enum_decl, None, ctx)
            }
            Statement::ExportNamedDeclaration(decl) => {
                let span = decl.span;
                if let Some(Declaration::TSEnumDeclaration(ts_enum_decl)) = &mut decl.declaration {
                    self.transform_ts_enum(ts_enum_decl, Some(span), ctx)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(new_stmt) = new_stmt {
            *stmt = new_stmt;
        }
    }

    fn enter_expression(&mut self, expr: &mut Expression<'a>, ctx: &mut TraverseCtx<'a>) {
        if let Expression::StaticMemberExpression(member_expr) = expr
            && let Some(value) = Self::try_inline_enum_member(member_expr, ctx)
        {
            // TODO: Attach a trailing block comment `/* EnumName.MemberName */` to the
            // inlined literal to match TypeScript/Babel behavior (e.g. `0 /* Direction.Up */`).
            // This requires support for synthetic comments with arbitrary text content,
            // which the current oxc comment infrastructure (source-text-span-based) does not
            // provide. Options:
            // 1. Add an `annotation: Option<Atom<'a>>` field to NumericLiteral/StringLiteral
            //    and print it as a trailing block comment in codegen.
            // 2. Extend the Comment system to support non-source-text content.
            let _enum_name = if let Expression::Identifier(ident) = &member_expr.object {
                Some(ident.name.as_str())
            } else {
                None
            };
            let _member_name = member_expr.property.name.as_str();

            *expr = match value {
                ConstantValue::Number(n) => Self::get_initializer_expr(n, ctx),
                ConstantValue::String(s) => {
                    ctx.ast.expression_string_literal(SPAN, ctx.ast.atom(&s), None)
                }
            };
        }
    }
}

impl<'a> TypeScriptEnum {
    /// ```TypeScript
    /// enum Foo {
    ///   X = 1,
    ///   Y
    /// }
    /// ```
    /// ```JavaScript
    /// var Foo = ((Foo) => {
    ///   Foo[Foo["X"] = 1] = "X";
    ///   Foo[Foo["Y"] = 2] = "Y";
    ///   return Foo;
    /// })(Foo || {});
    /// ```
    fn transform_ts_enum(
        &self,
        decl: &mut TSEnumDeclaration<'a>,
        export_span: Option<Span>,
        ctx: &mut TraverseCtx<'a>,
    ) -> Option<Statement<'a>> {
        if decl.declare {
            return None;
        }

        // Handle const enum optimization: remove declaration entirely
        // (references are inlined by enter_expression)
        if decl.r#const && self.optimize_const_enums {
            return None;
        }

        let ast = ctx.ast;

        let is_export = export_span.is_some();
        let is_not_top_scope = !ctx.scoping().scope_flags(ctx.current_scope_id()).is_top();

        let enum_name: Ident = decl.id.name;
        let func_scope_id = decl.body.scope_id();
        let param_binding =
            ctx.generate_binding(enum_name, func_scope_id, SymbolFlags::FunctionScopedVariable);

        let id = param_binding.create_binding_pattern(ctx);

        // ((Foo) => {
        let params =
            ast.formal_parameter(SPAN, ast.vec(), id, NONE, NONE, false, None, false, false);
        let params = ast.vec1(params);
        let params = ast.alloc_formal_parameters(
            SPAN,
            FormalParameterKind::ArrowFormalParameters,
            params,
            NONE,
        );

        let has_potential_side_effect = decl.body.members.iter().any(|member| {
            matches!(
                member.initializer,
                Some(Expression::NewExpression(_) | Expression::CallExpression(_))
            )
        });

        let statements = Self::transform_ts_enum_members(
            func_scope_id,
            &mut decl.body.members,
            &param_binding,
            ctx,
        );
        let span = decl.span;
        let body = ast.alloc_function_body(span, ast.vec(), statements);
        let callee = ctx.ast.expression_function_with_scope_id_and_pure_and_pife(
            span,
            FunctionType::FunctionExpression,
            None,
            false,
            false,
            false,
            NONE,
            NONE,
            params,
            NONE,
            Some(body),
            func_scope_id,
            false,
            false,
        );

        let enum_symbol_id = decl.id.symbol_id();

        // Foo[Foo["X"] = 0] = "X";
        let redeclarations = ctx.scoping().symbol_redeclarations(enum_symbol_id);
        let is_already_declared =
            redeclarations.first().map_or_else(|| false, |rd| rd.span != decl.id.span);

        let arguments = if (is_export || is_not_top_scope) && !is_already_declared {
            // }({});
            let object_expr = ast.expression_object(SPAN, ast.vec());
            ast.vec1(Argument::from(object_expr))
        } else {
            // }(Foo || {});
            let op = LogicalOperator::Or;
            let left = ctx.create_bound_ident_expr(
                decl.id.span,
                enum_name,
                enum_symbol_id,
                ReferenceFlags::Read,
            );
            let right = ast.expression_object(SPAN, ast.vec());
            let expression = ast.expression_logical(span, left, op, right);
            ast.vec1(Argument::from(expression))
        };

        let call_expression = ast.expression_call_with_pure(
            span,
            callee,
            NONE,
            arguments,
            false,
            !has_potential_side_effect,
        );

        if is_already_declared {
            let op = AssignmentOperator::Assign;
            let left = ctx.create_bound_ident_reference(
                decl.id.span,
                enum_name,
                enum_symbol_id,
                ReferenceFlags::Write,
            );
            let left = AssignmentTarget::AssignmentTargetIdentifier(ctx.alloc(left));
            let expr = ast.expression_assignment(span, op, left, call_expression);
            return Some(ast.statement_expression(span, expr));
        }

        let kind = if is_export || is_not_top_scope {
            VariableDeclarationKind::Let
        } else {
            VariableDeclarationKind::Var
        };
        let decls = {
            let binding_identifier = decl.id.clone();
            let binding = BindingPattern::BindingIdentifier(ctx.alloc(binding_identifier));
            let decl =
                ast.variable_declarator(span, kind, binding, NONE, Some(call_expression), false);
            ast.vec1(decl)
        };
        let variable_declaration = ast.declaration_variable(span, kind, decls, false);

        let stmt = if let Some(export_span) = export_span {
            let declaration = ctx
                .ast
                .plain_export_named_declaration_declaration(export_span, variable_declaration);
            Statement::ExportNamedDeclaration(declaration)
        } else {
            Statement::from(variable_declaration)
        };
        Some(stmt)
    }

    fn transform_ts_enum_members(
        enum_scope_id: ScopeId,
        members: &mut ArenaVec<'a, TSEnumMember<'a>>,
        param_binding: &BoundIdentifier<'a>,
        ctx: &mut TraverseCtx<'a>,
    ) -> ArenaVec<'a, Statement<'a>> {
        let ast = ctx.ast;

        let mut statements = ast.vec();

        // If enum number has no initializer, its value will be the previous member value + 1,
        // if it's the first member, it will be `0`.
        // It used to keep track of the previous constant number.
        let mut prev_constant_number = Some(-1.0);

        let mut prev_member_name = None;

        for member in members.take_in(ctx.ast) {
            let member_span = member.span;
            let member_name = member.id.static_name();

            let init = if let Some(mut initializer) = member.initializer {
                // Look up the pre-computed constant value from Scoping
                let constant_value: Option<ConstantValue> = ctx
                    .scoping()
                    .get_binding(enum_scope_id, member_name.as_str().into())
                    .and_then(|sym_id| ctx.scoping().get_enum_member_value(sym_id))
                    .cloned();

                match constant_value {
                    None => {
                        prev_constant_number = None;

                        IdentifierReferenceRename::new(param_binding.name, enum_scope_id, ctx)
                            .visit_expression(&mut initializer);

                        initializer
                    }
                    Some(constant_value) => match constant_value {
                        ConstantValue::Number(v) => {
                            prev_constant_number = Some(v);
                            Self::get_initializer_expr(v, ctx)
                        }
                        ConstantValue::String(s) => {
                            prev_constant_number = None;
                            ast.expression_string_literal(SPAN, ctx.ast.atom(&s), None)
                        }
                    },
                }
                // No initializer, try to infer the value from the previous member.
            } else if let Some(value) = &prev_constant_number {
                let value = value + 1.0;
                prev_constant_number = Some(value);
                Self::get_initializer_expr(value, ctx)
            } else if let Some(prev_member_name) = prev_member_name {
                let self_ref = {
                    let obj = param_binding.create_read_expression(ctx);
                    let expr = ctx.ast.expression_string_literal(SPAN, prev_member_name, None);
                    ast.member_expression_computed(SPAN, obj, expr, false).into()
                };

                // 1 + Foo["x"]
                let one = Self::get_number_literal_expression(1.0, ctx);
                ast.expression_binary(SPAN, one, BinaryOperator::Addition, self_ref)
            } else {
                Self::get_number_literal_expression(0.0, ctx)
            };

            let is_str = init.is_string_literal();

            // Foo["x"] = init
            let member_expr = {
                let obj = param_binding.create_read_expression(ctx);
                let expr = ast.expression_string_literal(SPAN, member_name, None);

                ast.member_expression_computed(SPAN, obj, expr, false)
            };
            let left = SimpleAssignmentTarget::from(member_expr);
            let mut expr = ast.expression_assignment(
                member_span,
                AssignmentOperator::Assign,
                left.into(),
                init,
            );

            // Foo[Foo["x"] = init] = "x"
            if !is_str {
                let member_expr = {
                    let obj = param_binding.create_read_expression(ctx);
                    ast.member_expression_computed(SPAN, obj, expr, false)
                };
                let left = SimpleAssignmentTarget::from(member_expr);
                let right = ast.expression_string_literal(SPAN, member_name, None);
                expr = ast.expression_assignment(
                    member_span,
                    AssignmentOperator::Assign,
                    left.into(),
                    right,
                );
            }

            prev_member_name = Some(member_name);
            statements.push(ast.statement_expression(member_span, expr));
        }

        let enum_ref = param_binding.create_read_expression(ctx);
        // return Foo;
        let return_stmt = ast.statement_return(SPAN, Some(enum_ref));
        statements.push(return_stmt);

        statements
    }

    fn get_number_literal_expression(value: f64, ctx: &TraverseCtx<'a>) -> Expression<'a> {
        ctx.ast.expression_numeric_literal(SPAN, value, None, NumberBase::Decimal)
    }

    fn get_initializer_expr(value: f64, ctx: &mut TraverseCtx<'a>) -> Expression<'a> {
        let is_negative = value < 0.0;

        // Infinity
        let expr = if value.is_infinite() {
            let infinity = ctx.ast.ident("Infinity");
            let infinity_symbol_id = ctx.scoping().find_binding(ctx.current_scope_id(), infinity);
            ctx.create_ident_expr(SPAN, infinity, infinity_symbol_id, ReferenceFlags::Read)
        } else {
            let value = if is_negative { -value } else { value };
            Self::get_number_literal_expression(value, ctx)
        };

        if is_negative {
            ctx.ast.expression_unary(SPAN, UnaryOperator::UnaryNegation, expr)
        } else {
            expr
        }
    }

    /// Emit a const enum declaration as `var X = {}` placeholder for bundler mode.
    /// Try to inline an enum member access like `Direction.Up` to its literal value.
    /// Works for both const and regular enums when the member value is known.
    fn try_inline_enum_member(
        expr: &StaticMemberExpression<'a>,
        ctx: &TraverseCtx<'a>,
    ) -> Option<ConstantValue> {
        let Expression::Identifier(ident) = &expr.object else { return None };
        let ref_id = ident.reference_id.get()?;
        let symbol_id = ctx.scoping().get_reference(ref_id).symbol_id()?;

        let flags = ctx.scoping().symbol_flags(symbol_id);
        if !flags.is_const_enum() && !flags.contains(SymbolFlags::RegularEnum) {
            return None;
        }

        let body_scope_id = ctx.scoping().get_enum_body_scope(symbol_id)?;
        let property_name = &expr.property.name;

        let member_symbol_id =
            ctx.scoping().get_binding(body_scope_id, property_name.as_str().into())?;
        ctx.scoping().get_enum_member_value(member_symbol_id).cloned()
    }
}

/// Rename the identifier references in the enum members to `enum_name.identifier`
/// ```ts
/// enum A {
///    a = 1,
///    b = a.toString(),
///    d = c,
/// }
/// ```
/// will be transformed to
/// ```ts
/// enum A {
///   a = 1,
///   b = A.a.toString(),
///   d = A.c,
/// }
/// ```
struct IdentifierReferenceRename<'a, 'ctx> {
    enum_name: Ident<'a>,
    enum_scope_id: ScopeId,
    scope_stack: NonEmptyStack<ScopeId>,
    ctx: &'ctx TraverseCtx<'a>,
}

impl<'a, 'ctx> IdentifierReferenceRename<'a, 'ctx> {
    fn new(enum_name: Ident<'a>, enum_scope_id: ScopeId, ctx: &'ctx TraverseCtx<'a>) -> Self {
        IdentifierReferenceRename {
            enum_name,
            enum_scope_id,
            scope_stack: NonEmptyStack::new(enum_scope_id),
            ctx,
        }
    }
}

impl IdentifierReferenceRename<'_, '_> {
    fn should_reference_enum_member(&self, ident: &IdentifierReference<'_>) -> bool {
        let scoping = self.ctx.scoping.scoping();

        // Don't need to rename the identifier if it's not an enum member in this scope.
        // Check both that the binding exists AND that it has the EnumMember flag,
        // because the IIFE parameter (same name as the enum) is also in this scope.
        let is_enum_member = scoping
            .get_binding(self.enum_scope_id, ident.name)
            .is_some_and(|sym_id| scoping.symbol_flags(sym_id).is_enum_member());
        if !is_enum_member {
            return false;
        }

        let Some(symbol_id) = scoping.get_reference(ident.reference_id()).symbol_id() else {
            // No symbol found, yet the name exists as a binding in the enum scope.
            // It must be referencing a member declared in a previous enum block: `enum Foo { A }; enum Foo { B = A }`
            return true;
        };

        let symbol_scope_id = scoping.symbol_scope_id(symbol_id);
        // Don't need to rename the identifier when it references a nested enum member:
        //
        // ```ts
        // enum OuterEnum {
        //   A = 0,
        //   B = () => {
        //     enum InnerEnum {
        //       A = 0,
        //       B = A,
        //           ^ This references to `InnerEnum.A` should not be renamed
        //     }
        //     return InnerEnum.B;
        //   }
        // }
        // ```
        *self.scope_stack.first() == symbol_scope_id
            // The resolved symbol is declared outside the enum,
            // and we have checked that the name exists as a binding in the enum scope:
            //
            // ```ts
            // const A = 0;
            // enum Foo { A }
            // enum Foo { B = A }
            //                ^ This should be renamed to Foo.A
            // ```
            || !self.scope_stack.contains(&symbol_scope_id)
    }
}

impl<'a> VisitMut<'a> for IdentifierReferenceRename<'a, '_> {
    fn enter_scope(&mut self, _flags: ScopeFlags, scope_id: &Cell<Option<ScopeId>>) {
        self.scope_stack.push(scope_id.get().unwrap());
    }

    fn leave_scope(&mut self) {
        self.scope_stack.pop();
    }

    fn visit_expression(&mut self, expr: &mut Expression<'a>) {
        match expr {
            Expression::Identifier(ident) if self.should_reference_enum_member(ident) => {
                let object = self.ctx.ast.expression_identifier(SPAN, self.enum_name);
                let property = self.ctx.ast.identifier_name(SPAN, ident.name);
                *expr = self.ctx.ast.member_expression_static(SPAN, object, property, false).into();
            }
            _ => {
                walk_mut::walk_expression(self, expr);
            }
        }
    }
}
