use serde::{Deserialize, Serialize};

/// The Canonical JSON format for Satellite (.jsonc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalCnf {
    pub version: String,
    pub vars: Vec<VarDef>,
    #[serde(default)]
    pub imports: Vec<ImportDef>,
    #[serde(default)]
    pub functions: Vec<FunctionDef>,
    pub clauses: Vec<ClauseDef>,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarDef {
    pub name: String,
    #[serde(rename = "type")]
    pub var_type: String, // "bool", "batch", "vec"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim1: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dim2: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDef {
    pub path: String,
    pub hash: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    pub signature: String,
    #[serde(default)]
    pub auto_generated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseDef {
    pub expr: String,
    pub id: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub style: String,
    pub source_hash: String,
}

impl CanonicalCnf {
    pub fn from_reader<R: std::io::Read>(reader: R) -> serde_json::Result<Self> {
        serde_json::from_reader(reader)
    }

    pub fn to_writer<W: std::io::Write>(&self, writer: W) -> serde_json::Result<()> {
        serde_json::to_writer_pretty(writer, self)
    }
}

use crate::advanced_cnf::{AdvancedCnf, VariableDef, Clause, ClauseType, AbiConstraint};
use crate::parser::{Parser, Expr};
use satellite_base::types::VarType;
use std::collections::HashMap;

impl TryFrom<CanonicalCnf> for AdvancedCnf {
    type Error = String;

    fn try_from(canonical: CanonicalCnf) -> Result<Self, Self::Error> {
        let mut acnf = AdvancedCnf::new();
        let mut name_to_id = HashMap::new();
        let mut next_id = 1;

        // 1. Process variables
        for var in canonical.vars {
            let id = next_id;
            name_to_id.insert(var.name.clone(), id);
            
            let vtype = match var.var_type.as_str() {
                "bool" => VarType::Bool,
                "batch" => VarType::Batch { 
                    dim: var.dim1.ok_or("Batch missing dim1")? 
                },
                "vec" => VarType::Vec {
                    inner_dim: var.dim1.ok_or("Vec missing dim1")?,
                    outer_dim: var.dim2.ok_or("Vec missing dim2")?,
                },
                _ => return Err(format!("Unknown var type: {}", var.var_type)),
            };

            acnf.variables.push(VariableDef {
                id,
                var_type: vtype.clone(),
                name: Some(var.name),
            });
            
            next_id += vtype.bool_count() as u64;
        }

        // 2. Process clauses (expressions)
        for clause_def in canonical.clauses {
            let mut parser = Parser::new(&clause_def.expr);
            let expr = parser.parse().map_err(|e| format!("Parse error in clause {}: {}", clause_def.id, e))?;
            
            // Lower expression to CNF clauses or AbiConstraint
            lower_expr(expr, &name_to_id, &mut acnf)?;
        }

        Ok(acnf)
    }
}

fn lower_expr(expr: Expr, map: &HashMap<String, u64>, acnf: &mut AdvancedCnf) -> Result<(), String> {
    match expr {
        Expr::BinaryOp { op, left, right } => {
            if op == "AND" || op == "and" || op == "&&" {
                // Conjunction: Split
                lower_expr(*left, map, acnf)?;
                lower_expr(*right, map, acnf)?;
            } else if op == "OR" || op == "or" || op == "||" {
                // Disjunction: Create clause (simplistic CNF only support for now)
                // We assume strict CNF structure for standard clauses: OR of Literals
                // Handling (A AND B) OR C requires distribution which we skip for MVP
                let mut literals = Vec::new();
                collect_disjunction(*left, map, &mut literals)?;
                collect_disjunction(*right, map, &mut literals)?;
                acnf.clauses.push(Clause {
                    literals,
                    clause_type: ClauseType::Original,
                    lbd: None
                });
            } else if op == "eq" {
                 // Arithmetic/Constraint
                 // Treat as ABI constraint
                 // Collect variables used
                 let inputs = collect_vars_deep(&Expr::BinaryOp{op: op.clone(), left: left.clone(), right: right.clone()}, map);
                 acnf.abi_constraints.push(AbiConstraint {
                     name: "eq".to_string(), // Or generate a specific name based on expression
                     inputs,
                     code_hash: "interactive_eq".to_string(), // Placeholder
                     cached: false,
                 });
            } else {
                 return Err(format!("Unsupported binary op at top level: {}", op));
            }
        },
        Expr::UnaryOp { .. } | Expr::Var(_) | Expr::Lit(_) => {
             // Single literal clause
             let mut literals = Vec::new();
             collect_disjunction(expr, map, &mut literals)?;
             acnf.clauses.push(Clause {
                literals,
                clause_type: ClauseType::Original,
                lbd: None
             });
        }
        _ => return Err("Unsupported top level expression".to_string())
    }
    Ok(())
}

fn collect_disjunction(expr: Expr, map: &HashMap<String, u64>, literals: &mut Vec<i64>) -> Result<(), String> {
    match expr {
        Expr::BinaryOp { op, left, right } => {
            if op == "OR" || op == "or" || op == "||" {
                collect_disjunction(*left, map, literals)?;
                collect_disjunction(*right, map, literals)?;
            } else {
                return Err("Nested non-OR in disjunction not supported in MVP".to_string()); // Correct distribution needed here for full support
            }
        },
        Expr::Var(name) => {
            if let Some(&id) = map.get(&name) {
                // Remove % prefix if present in name? Parser handles % in parse_var, returning var name with %.
                // Wait, parser: Token::Var includes %.
                // Map keys include %.
                 literals.push(id as i64);
            } else {
                 // Try removing %
                 if let Some(&id) = map.get(name.trim_start_matches('%')) {
                     literals.push(id as i64);
                 } else {
                     return Err(format!("Unknown variable: {}", name));
                 }
            }
        },
        Expr::UnaryOp { op, expr } => {
            if op == "NOT" || op == "not" || op == "!" || op == "-" {
                 if let Expr::Var(name) = *expr {
                     let id = if let Some(&id) = map.get(&name) { id }
                     else if let Some(&id) = map.get(name.trim_start_matches('%')) { id }
                     else { return Err(format!("Unknown variable: {}", name)); };
                     literals.push(-(id as i64));
                 } else {
                     return Err("Only direct negation of vars supported in MVP".to_string());
                 }
            } else {
                return Err(format!("Unknown unary op: {}", op));
            }
        }
        _ => return Err("Unsupported element in disjunction".to_string())
    }
    Ok(())
}

fn collect_vars_deep(expr: &Expr, map: &HashMap<String, u64>) -> Vec<u64> {
     let mut vars = Vec::new();
     match expr {
         Expr::Var(name) => {
             if let Some(&id) = map.get(name) { vars.push(id); }
             else if let Some(&id) = map.get(name.trim_start_matches('%')) { vars.push(id); }
         },
         Expr::BinaryOp { left, right, .. } => {
             vars.extend(collect_vars_deep(left, map));
             vars.extend(collect_vars_deep(right, map));
         },
         Expr::UnaryOp { expr, .. } => {
             vars.extend(collect_vars_deep(expr, map));
         },
         Expr::Call { args, .. } => {
             for arg in args {
                 vars.extend(collect_vars_deep(arg, map));
             }
         }
         _ => {}
     }
     vars
}
