//! Comprehensive unit tests for satellite-cdcl crate.

use satellite_cdcl::bcp::{PropagationQueue, WatchedLiterals};
use satellite_cdcl::conflict::{ConflictAnalyzer, ImplicationGraph};
use satellite_cdcl::heuristics::{EvsidsScores, VsidsScores};
use satellite_cdcl::{CdclSolver, SatResult};
use satellite_format::advanced_cnf::{AdvancedCnf, Clause, ClauseType};

// =============================================================================
// Propagation Queue Tests
// =============================================================================

#[test]
fn test_prop_queue_basic() {
    let mut queue = PropagationQueue::new();
    assert!(queue.is_empty());
    
    queue.enqueue(1);
    queue.enqueue(-2);
    queue.enqueue(3);
    
    assert!(!queue.is_empty());
    assert_eq!(queue.dequeue(), Some(1));
    assert_eq!(queue.dequeue(), Some(-2));
    assert_eq!(queue.dequeue(), Some(3));
    assert_eq!(queue.dequeue(), None);
    assert!(queue.is_empty());
}

#[test]
fn test_prop_queue_clear() {
    let mut queue = PropagationQueue::new();
    queue.enqueue(1);
    queue.enqueue(2);
    queue.clear();
    
    assert!(queue.is_empty());
    assert_eq!(queue.dequeue(), None);
}

// =============================================================================
// Watched Literals Tests
// =============================================================================

#[test]
fn test_watched_literals_add_and_get() {
    let mut watches = WatchedLiterals::new(10);
    
    watches.add_watch(1, 0, 2);
    watches.add_watch(1, 1, -3);
    watches.add_watch(-1, 2, 4);
    
    let pos_watches = watches.get_watches(1);
    assert_eq!(pos_watches.len(), 2);
    assert_eq!(pos_watches[0].clause_id, 0);
    assert_eq!(pos_watches[0].blocker, 2);
    
    let neg_watches = watches.get_watches(-1);
    assert_eq!(neg_watches.len(), 1);
    assert_eq!(neg_watches[0].clause_id, 2);
}

#[test]
fn test_watched_literals_empty() {
    let watches = WatchedLiterals::new(10);
    assert!(watches.get_watches(1).is_empty());
    assert!(watches.get_watches(-5).is_empty());
}

// =============================================================================
// VSIDS Heuristics Tests
// =============================================================================

#[test]
fn test_vsids_bump_and_decay() {
    let mut vsids = VsidsScores::new(10);
    
    assert_eq!(vsids.score(0), 0.0);
    assert_eq!(vsids.score(5), 0.0);
    
    vsids.bump(2);
    assert!(vsids.score(2) > 0.0);
    
    vsids.bump(2);
    let score_before_decay = vsids.score(2);
    
    vsids.decay();
    vsids.bump(2);
    let score_after = vsids.score(2);
    assert!(score_after > score_before_decay);
}

#[test]
fn test_evsids_scoring() {
    let mut evsids = EvsidsScores::new(10);
    
    evsids.bump(3);
    evsids.on_conflict();
    evsids.bump(3);
    
    assert!(evsids.score(3) > 0.0);
    assert_eq!(evsids.score(0), 0.0);
}

// =============================================================================
// Implication Graph Tests
// =============================================================================

#[test]
fn test_implication_graph_basic() {
    let mut graph = ImplicationGraph::new(10);
    
    graph.assign(0, 1, None);
    graph.assign(1, 1, Some(0));
    graph.assign(2, 1, Some(1));
    graph.assign(3, 2, None);
    
    assert_eq!(graph.level(0), 1);
    assert_eq!(graph.level(1), 1);
    assert_eq!(graph.level(3), 2);
    
    assert_eq!(graph.reason(0), None);
    assert_eq!(graph.reason(1), Some(0));
}

#[test]
fn test_implication_graph_backtrack() {
    let mut graph = ImplicationGraph::new(10);
    
    graph.assign(0, 1, None);
    graph.assign(1, 1, Some(0));
    graph.assign(2, 2, None);
    graph.assign(3, 2, Some(1));
    
    graph.backtrack(1);
    
    assert_eq!(graph.reason(2), None);
    assert_eq!(graph.reason(3), None);
}

// =============================================================================
// Conflict Analyzer Tests
// =============================================================================

#[test]
fn test_conflict_analyzer_lbd() {
    let analyzer = ConflictAnalyzer::new(10);
    let graph = {
        let mut g = ImplicationGraph::new(10);
        g.assign(0, 1, None);
        g.assign(1, 2, None);
        g.assign(2, 2, Some(0));
        g.assign(3, 3, None);
        g
    };
    
    let clause = vec![1, -2, 3, -4];
    let lbd = analyzer.compute_lbd(&clause, &graph);
    assert_eq!(lbd, 3);
}

// =============================================================================
// Full Solver Integration Tests
// =============================================================================

fn make_clause(lits: Vec<i64>) -> Clause {
    Clause {
        literals: lits,
        clause_type: ClauseType::Original,
        lbd: None,
    }
}

#[test]
fn test_solver_empty_problem() {
    let problem = AdvancedCnf::new();
    
    let mut solver = CdclSolver::new(&problem);
    let result = solver.solve().unwrap();
    
    match result {
        SatResult::Sat(_) => (),
        _ => panic!("Empty problem should be SAT"),
    }
}

#[test]
fn test_solver_single_unit_clause() {
    let mut problem = AdvancedCnf::new();
    problem.clauses.push(make_clause(vec![1]));
    
    let mut solver = CdclSolver::new(&problem);
    let result = solver.solve().unwrap();
    
    match result {
        SatResult::Sat(model) => {
            assert_eq!(model.get(0), Some(&true));
        }
        _ => panic!("Single unit clause should be SAT"),
    }
}

#[test]
fn test_solver_conflicting_units() {
    let mut problem = AdvancedCnf::new();
    problem.clauses.push(make_clause(vec![1]));
    problem.clauses.push(make_clause(vec![-1]));
    
    let mut solver = CdclSolver::new(&problem);
    let result = solver.solve().unwrap();
    
    match result {
        SatResult::Unsat => (),
        _ => panic!("Conflicting units should be UNSAT"),
    }
}

#[test]
fn test_solver_simple_sat() {
    let mut problem = AdvancedCnf::new();
    problem.clauses.push(make_clause(vec![1, 2]));
    problem.clauses.push(make_clause(vec![-1, 3]));
    problem.clauses.push(make_clause(vec![-2, -3]));
    
    let mut solver = CdclSolver::new(&problem);
    let result = solver.solve().unwrap();
    
    match result {
        SatResult::Sat(model) => {
            let x1 = model.get(0).copied().unwrap_or(false);
            let x2 = model.get(1).copied().unwrap_or(false);
            let x3 = model.get(2).copied().unwrap_or(false);
            
            assert!(x1 || x2, "Clause 1 not satisfied");
            assert!(!x1 || x3, "Clause 2 not satisfied");
            assert!(!x2 || !x3, "Clause 3 not satisfied");
        }
        _ => panic!("Expected SAT"),
    }
}

#[test]
fn test_solver_pigeonhole_2_1() {
    let mut problem = AdvancedCnf::new();
    problem.clauses.push(make_clause(vec![1]));
    problem.clauses.push(make_clause(vec![2]));
    problem.clauses.push(make_clause(vec![-1, -2]));
    
    let mut solver = CdclSolver::new(&problem);
    let result = solver.solve().unwrap();
    
    match result {
        SatResult::Unsat => (),
        _ => panic!("Pigeonhole 2-1 should be UNSAT"),
    }
}
