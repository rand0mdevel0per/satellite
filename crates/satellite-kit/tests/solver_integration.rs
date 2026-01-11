//! End-to-end integration tests for satellite-kit solver.

use satellite_kit::Solver;
use satellite_cdcl::SatResult;

/// Test basic SAT problem solving with raw DIMACS-style literals.
#[test]
fn test_simple_sat_dimacs() {
    let mut solver = Solver::new();
    
    // Use DIMACS format: positive = true, negative = false
    // Variables: x1=1, x2=2, x3=3 (1-indexed in DIMACS, 0-indexed in model)
    
    // Clause: (x1 OR NOT x2) = (1 OR -2)
    solver.add_clause([1, -2]);
    
    // Clause: (x2 OR x3) = (2 OR 3)
    solver.add_clause([2, 3]);
    
    // Clause: (NOT x1 OR NOT x3) = (-1 OR -3)
    solver.add_clause([-1, -3]);
    
    let result = solver.solve().expect("Solve should not error");
    
    match result {
        SatResult::Sat(model) => {
            // Model is 0-indexed, so var 1 (DIMACS) = index 0
            let v1 = model.get(0).copied().unwrap_or(false);
            let v2 = model.get(1).copied().unwrap_or(false);
            let v3 = model.get(2).copied().unwrap_or(false);
            
            // (x1 OR NOT x2)
            assert!(v1 || !v2, "Clause 1 not satisfied");
            // (x2 OR x3)
            assert!(v2 || v3, "Clause 2 not satisfied");
            // (NOT x1 OR NOT x3)
            assert!(!v1 || !v3, "Clause 3 not satisfied");
        }
        other => panic!("Expected SAT, got {:?}", other),
    }
}

/// Test UNSAT detection.
#[test]
fn test_simple_unsat() {
    let mut solver = Solver::new();
    
    // x AND NOT x = UNSAT
    solver.add_clause([1]);   // x1 = true
    solver.add_clause([-1]);  // x1 = false
    
    let result = solver.solve().expect("Solve should not error");
    
    match result {
        SatResult::Unsat => (), // Expected
        other => panic!("Expected UNSAT, got {:?}", other),
    }
}

/// Test empty problem (trivially SAT).
#[test]
fn test_empty_problem() {
    let mut solver = Solver::new();
    let result = solver.solve().expect("Solve should not error");
    
    match result {
        SatResult::Sat(_) => (), // Expected
        other => panic!("Expected SAT for empty problem, got {:?}", other),
    }
}

/// Test single unit clause.
#[test]
fn test_unit_clause() {
    let mut solver = Solver::new();
    
    // Unit clause: x1 must be true
    solver.add_clause([1]);
    
    // (x1 OR x2)
    solver.add_clause([1, 2]);
    
    let result = solver.solve().expect("Solve should not error");
    
    match result {
        SatResult::Sat(model) => {
            // x1 (index 0) should be true due to unit propagation
            assert_eq!(model.get(0), Some(&true), "x1 should be true");
        }
        other => panic!("Expected SAT, got {:?}", other),
    }
}

/// Test pigeonhole principle (UNSAT for n+1 pigeons in n holes).
#[test]
fn test_pigeonhole_unsat() {
    // 3 pigeons, 2 holes (UNSAT)
    let mut solver = Solver::new();
    
    // Variable naming: pij = pigeon i in hole j
    // p11=1, p12=2, p21=3, p22=4, p31=5, p32=6
    
    // Each pigeon must be in at least one hole
    solver.add_clause([1, 2]);   // p1 in h1 or h2
    solver.add_clause([3, 4]);   // p2 in h1 or h2
    solver.add_clause([5, 6]);   // p3 in h1 or h2
    
    // Each hole can hold at most one pigeon
    // Hole 1: at most one of p11, p21, p31 (vars 1, 3, 5)
    solver.add_clause([-1, -3]); // not (p1h1 and p2h1)
    solver.add_clause([-1, -5]); // not (p1h1 and p3h1)
    solver.add_clause([-3, -5]); // not (p2h1 and p3h1)
    
    // Hole 2: at most one of p12, p22, p32 (vars 2, 4, 6)
    solver.add_clause([-2, -4]); // not (p1h2 and p2h2)
    solver.add_clause([-2, -6]); // not (p1h2 and p3h2)
    solver.add_clause([-4, -6]); // not (p2h2 and p3h2)
    
    let result = solver.solve().expect("Solve should not error");
    
    match result {
        SatResult::Unsat => (), // Expected - 3 pigeons can't fit in 2 holes
        other => panic!("Expected UNSAT for pigeonhole, got {:?}", other),
    }
}
