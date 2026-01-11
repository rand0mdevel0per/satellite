use crate::parser::{Expr, Parser, Token, Tokenizer};
use crate::canonical::{CanonicalCnf, Clause, Var, VarType};
use std::collections::HashMap;

pub struct SatParser<'a> {
    input: &'a str,
}

impl<'a> SatParser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self { input }
    }

    pub fn parse(&self) -> Result<CanonicalCnf, String> {
        // We need to parse a sequence of statements.
        // Statement: (Var =)? Expr
        // But our existing Parser expects EOF after one expression.
        // We should reuse Tokenizer to get tokens, identifying boundaries?
        // Or simply split by newlines? Newlines might be part of expression?
        // The tokenizer skips whitespace including newlines.
        
        // Better approach: Use the Tokenizer to stream tokens.
        // We look for "Var =" pattern or just "Expr".
        
        let mut tokenizer = Tokenizer::new(self.input);
        let mut clauses = Vec::new();
        let mut vars_map: HashMap<String, VarType> = HashMap::new();
        let mut next_clause_id = 0;

        // Custom loop over tokens
        // We can't reuse Parser for multiple statements easily because it consumes ownership or has fixed state.
        // Let's create a Parser for chunks?
        
        // Actually, the syntax in specs allows:
        // o = %x7 and %x6
        //   AND %x5 ...
        // This implies one big expression that continues until... EOF? Or semicolon?
        // The specs didn't specify a terminator.
        // Assuming for now the file contains ONE constraint expression OR a list of clauses.
        // If the user said "Solve .sat file", and provided the Sudoku example, it looks like ONE expression potentially broken into lines.
        
        // Let's try to parse ONE expression from the whole file first.
        let mut parser = Parser::new(self.input);
        let expr = parser.parse()?;
        
        // Now convert Expr to canonical string form for the clause.
        // And traverse Expr to find variables.
        
        // The parser returns an Expr AST.
        // We need to flatten it back to string for CanonicalCnf?
        // CanonicalCnf expects "expr" string field in Clause.
        // It uses strings like "%x7 AND %x6".
        
        // We can just gather the variables from the Expr.
        self.collect_vars(&expr, &mut vars_map);
        
        let clause_str = self.expr_to_string(&expr);
        
        clauses.push(Clause {
            id: 0,
            expr: clause_str,
        });
        
        // Convert vars_map to Vec<Var>
        let vars = vars_map.into_iter().map(|(name, vtype)| {
            Var {
                name,
                r#type: vtype, // Determine type?
            }
        }).collect();

        Ok(CanonicalCnf {
            version: "1.0".to_string(),
            vars,
            imports: vec![],
            functions: vec![],
            clauses,
            metadata: HashMap::new(),
        })
    }
    
    fn collect_vars(&self, expr: &Expr, vars: &mut HashMap<String, VarType>) {
        match expr {
            Expr::Var(name) => {
                // Infer type based on prefix?
                // % -> Bool
                // %^ -> Batch
                // %* -> Vec
                if !vars.contains_key(name) {
                    let vtype = if name.contains("^") {
                         VarType::Batch { dim: 0 } // Unknown dim
                    } else if name.contains("*") {
                         VarType::Vec { dim1: 0, dim2: 0 }
                    } else {
                         VarType::Bool
                    };
                     // Clean name? extract after %? 
                     // The Canonical format expects names like "x7", "x6".
                     // The parser returns "%x7". 
                     // We should store "x7" in vars, but keep "%x7" in expr?
                     // Canonical logic: parser.rs treats %... as Var(String).
                     // When lowering, it expects %name.
                     
                     // CanonicalCnf vars should be just "name".
                     // So we strip % and store.
                     let clean_name = name.trim_start_matches('%').trim_start_matches('^').trim_start_matches('*').to_string();
                     vars.insert(clean_name, vtype);
                }
            }
            Expr::BinaryOp { left, right, .. } => {
                self.collect_vars(left, vars);
                self.collect_vars(right, vars);
            }
            Expr::UnaryOp { expr, .. } => {
                self.collect_vars(expr, vars);
            }
            Expr::Call { args, .. } => {
                for arg in args {
                    self.collect_vars(arg, vars);
                }
            }
            _ => {}
        }
    }
    
    fn expr_to_string(&self, expr: &Expr) -> String {
        // Reconstruct string. Ideally close to input but normalized?
        match expr {
            Expr::Var(s) => s.clone(),
            Expr::Lit(i) => i.to_string(),
            Expr::BinaryOp { op, left, right } => {
                format!("({} {} {})", self.expr_to_string(left), op, self.expr_to_string(right))
            }
            Expr::UnaryOp { op, expr } => {
                format!("{} {}", op, self.expr_to_string(expr))
            }
            Expr::Call { name, args } => {
                let args_str: Vec<String> = args.iter().map(|a| self.expr_to_string(a)).collect();
                format!("{}({})", name, args_str.join(", "))
            }
        }
    }
}
