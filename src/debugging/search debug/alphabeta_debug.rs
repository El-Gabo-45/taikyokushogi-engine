use crate::board::Board;
use crate::eval::evaluate;
use crate::types::{BLACK, WHITE};

/// Snapshot útil para trazar una llamada de alpha-beta.
#[derive(Debug, Clone)]
pub struct AlphaBetaDebugFrame {
    pub depth: u32,
    pub ply: u32,
    pub alpha: i32,
    pub beta: i32,
    pub in_check: bool,
    pub static_eval: i32,
    pub move_count: usize,
    pub node_count: u64,
    pub side_to_move: &'static str,
}

/// Construye un resumen corto para inspección de la rama actual.
pub fn debug_alphabeta_frame(
    board: &Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    ply: u32,
    in_check: bool,
    move_count: usize,
    node_count: u64,
) -> AlphaBetaDebugFrame {
    let static_eval = evaluate(board);
    let side_to_move = if board.side_to_move == BLACK { "black" } else { "white" };

    AlphaBetaDebugFrame {
        depth,
        ply,
        alpha,
        beta,
        in_check,
        static_eval,
        move_count,
        node_count,
        side_to_move,
    }
}

/// Formato legible para log o depuración interactiva.
pub fn format_alphabeta_debug(
    board: &Board,
    depth: u32,
    alpha: i32,
    beta: i32,
    ply: u32,
    in_check: bool,
    move_count: usize,
    node_count: u64,
) -> String {
    let frame = debug_alphabeta_frame(board, depth, alpha, beta, ply, in_check, move_count, node_count);
    format!(
        "alphabeta depth={} ply={} side={} alpha={} beta={} in_check={} eval={} moves={} nodes={}",
        frame.depth,
        frame.ply,
        frame.side_to_move,
        frame.alpha,
        frame.beta,
        frame.in_check,
        frame.static_eval,
        frame.move_count,
        frame.node_count,
    )
}

/// Mensaje corto para reportar una poda.
pub fn debug_prune(reason: &str, depth: u32, alpha: i32, beta: i32) -> String {
    format!("prune reason={reason} depth={depth} alpha={alpha} beta={beta}")
}
