use std::iter::Peekable;
use std::str::Chars;

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Var(String), // %name or %^name or %*name
    Int(i64),
    Ident(String), // AND, OR, eq, etc.
    Symbol(char), // +, -, *, /, etc.
    LParen,
    RParen,
    EOF,
}

pub struct Tokenizer<'a> {
    input: Peekable<Chars<'a>>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input: input.chars().peekable(),
        }
    }

    pub fn next_token(&mut self) -> Result<Token, String> {
        self.skip_whitespace();
        
        match self.input.peek() {
            None => Ok(Token::EOF),
            Some(&c) => match c {
                '(' => { self.input.next(); Ok(Token::LParen) }
                ')' => { self.input.next(); Ok(Token::RParen) }
                '+' | '-' | '*' | '/' | '^' | '!' | '&' | '|' => { 
                     // Check for multi-char ops like &&, ||, == could be handled here
                     // For now simple single char symbols
                     self.input.next();
                     Ok(Token::Symbol(c))
                }
                '%' => self.parse_var(),
                '0'..='9' => self.parse_int(),
                'a'..='z' | 'A'..='Z' | '_' => self.parse_ident(),
                _ => Err(format!("Unexpected character: {}", c)),
            }
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.input.peek() {
            if c.is_whitespace() {
                self.input.next();
            } else {
                break;
            }
        }
    }

    fn parse_var(&mut self) -> Result<Token, String> {
        let mut s = String::new();
        // Consume '%'
        s.push(self.input.next().unwrap());
        
        // Check for Prefix '^' or '*'
        if let Some(&c) = self.input.peek() {
             if c == '^' || c == '*' {
                 s.push(self.input.next().unwrap());
             }
        }

        while let Some(&c) = self.input.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(self.input.next().unwrap());
            } else {
                break;
            }
        }
        Ok(Token::Var(s))
    }

    fn parse_int(&mut self) -> Result<Token, String> {
        let mut s = String::new();
        while let Some(&c) = self.input.peek() {
             if c.is_digit(10) {
                 s.push(self.input.next().unwrap());
             } else {
                 break;
             }
        }
        let val = s.parse::<i64>().map_err(|_| "Invalid integer".to_string())?;
        Ok(Token::Int(val))
    }

    fn parse_ident(&mut self) -> Result<Token, String> {
        let mut s = String::new();
        while let Some(&c) = self.input.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(self.input.next().unwrap());
            } else {
                break;
            }
        }
        Ok(Token::Ident(s))
    }
}

// AST
#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    Lit(i64),
    BinaryOp { op: String, left: Box<Expr>, right: Box<Expr> },
    UnaryOp { op: String, expr: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
}

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    current_token: Token,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut tokenizer = Tokenizer::new(input);
        let current_token = tokenizer.next_token().unwrap_or(Token::EOF);
        Self { tokenizer, current_token }
    }

    fn advance(&mut self) {
        self.current_token = self.tokenizer.next_token().unwrap_or(Token::EOF);
    }

    pub fn parse(&mut self) -> Result<Expr, String> {
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;

        while let Token::Ident(ref op) = self.current_token {
            if op == "OR" || op == "or" {
                let op_str = op.clone();
                self.advance();
                let right = self.parse_and()?;
                left = Expr::BinaryOp { op: op_str, left: Box::new(left), right: Box::new(right) };
            } else {
                break;
            }
        }
        // Handle || symbol if we add it
        if let Token::Symbol('|') = self.current_token {
             // Expect another |
             self.advance();
             if let Token::Symbol('|') = self.current_token {
                 self.advance();
                 let right = self.parse_and()?;
                 left = Expr::BinaryOp { op: "OR".to_string(), left: Box::new(left), right: Box::new(right) };
             }
        }
        
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_equality()?;

        while let Token::Ident(ref op) = self.current_token {
            if op == "AND" || op == "and" {
                let op_str = op.clone();
                self.advance();
                let right = self.parse_equality()?;
                left = Expr::BinaryOp { op: op_str, left: Box::new(left), right: Box::new(right) };
            } else {
                break;
            }
        }
        // Handle && symbol
        if let Token::Symbol('&') = self.current_token {
             self.advance();
             if let Token::Symbol('&') = self.current_token {
                 self.advance();
                 let right = self.parse_equality()?;
                 left = Expr::BinaryOp { op: "AND".to_string(), left: Box::new(left), right: Box::new(right) };
             }
        }
        
        Ok(left)
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_additive()?;
        
        loop {
            let op = match &self.current_token {
                Token::Ident(s) if s == "eq" || s == "neq" => Some(s.clone()),
                Token::Symbol('=') => {
                    // Check for ==
                    // For now just assume = is equality or consume next =
                    Some("eq".to_string())
                },
                _ => None
            };

            if let Some(op_str) = op {
                self.advance();
                // Consume second = if present
                if op_str == "eq" {
                    if let Token::Symbol('=') = self.current_token { self.advance(); }
                }

                let right = self.parse_additive()?;
                left = Expr::BinaryOp { op: op_str, left: Box::new(left), right: Box::new(right) };
            } else {
                break;
            }
        }
        Ok(left)
    }
    
    fn parse_additive(&mut self) -> Result<Expr, String> {
         let mut left = self.parse_primary()?;
         
         loop {
             let op = match &self.current_token {
                 Token::Symbol('+') => Some("+"),
                 Token::Symbol('-') => Some("-"),
                 _ => None,
             };
             
             if let Some(o) = op {
                 let op_str = o.to_string();
                 self.advance();
                 let right = self.parse_primary()?;
                 left = Expr::BinaryOp { op: op_str, left: Box::new(left), right: Box::new(right) };
             } else {
                 break;
             }
         }
         Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.current_token.clone() {
            Token::Var(s) => {
                self.advance();
                Ok(Expr::Var(s))
            }
            Token::Int(i) => {
                self.advance();
                Ok(Expr::Lit(i))
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                if let Token::RParen = self.current_token {
                    self.advance();
                    Ok(expr)
                } else {
                    Err("Expected ')'".to_string())
                }
            }
            Token::Ident(s) => {
                // Could be function call or NOT
                if s == "NOT" || s == "not" {
                    self.advance();
                    let expr = self.parse_primary()?;
                    Ok(Expr::UnaryOp { op: "NOT".to_string(), expr: Box::new(expr) })
                } else {
                    self.advance();
                    // Check for call
                    if let Token::LParen = self.current_token {
                        self.advance();
                        let mut args = Vec::new();
                        if let Token::RParen = self.current_token {
                            self.advance();
                        } else {
                            loop {
                                args.push(self.parse_expr()?);
                                if let Token::Symbol(',') = self.current_token {
                                    self.advance();
                                } else {
                                    break;
                                }
                            }
                            if let Token::RParen = self.current_token {
                                self.advance();
                            } else {
                                return Err("Expected ')'".to_string());
                            }
                        }
                        Ok(Expr::Call { name: s, args })
                    } else {
                        // Just an ident, maybe a boolean constant?
                        if s == "True" || s == "true" { Ok(Expr::Lit(1)) }
                        else if s == "False" || s == "false" { Ok(Expr::Lit(0)) }
                        else { Err(format!("Unknown identifier: {}", s)) }
                    }
                }
            }
            Token::Symbol('-') | Token::Symbol('!') => {
                let op = if let Token::Symbol('-') = self.current_token { "-" } else { "!" };
                self.advance();
                let expr = self.parse_primary()?;
                Ok(Expr::UnaryOp { op: op.to_string(), expr: Box::new(expr) })
            }
            _ => Err(format!("Unexpected token: {:?}", self.current_token)),
        }
    }
}
