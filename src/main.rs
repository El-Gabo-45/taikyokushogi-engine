//! Taikyoku Shogi HTTP Server + Web GUI
//!
//! Serves a REST API and static files for the web frontend.
//! Run with: cargo run --bin taikyokushogi-server

use axum::{
    extract::{Path, Query, State},
    http::{HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use taikyokushogi::{Board, Color, GameResult};

// ============================================================
// State
// ============================================================
struct AppState {
    board: Mutex<GameState>,
}

struct GameState {
    board: Board,
    move_log: Vec<String>,
    score_history: Vec<i32>,
    half_move: u32,
    game_over: bool,
}

impl GameState {
    fn new() -> Self {
        let board = Board::initial();
        let score = board.material_score();
        GameState {
            board,
            move_log: Vec::new(),
            score_history: vec![score],
            half_move: 0,
            game_over: false,
        }
    }

    fn reset(&mut self) {
        *self = GameState::new();
    }

    fn record_move(&mut self, side: &str, piece: &str, fr: usize, fc: usize, tr: usize, tc: usize, promotion: bool) {
        self.half_move += 1;
        let promo_s = if promotion { "+" } else { "" };
        let entry = format!("{}. {}: {} ({},{})-({},{}){}", self.half_move, side, piece, fr, fc, tr, tc, promo_s);
        self.move_log.push(entry);
        let score = self.board.material_score();
        self.score_history.push(score);
    }
}

// ============================================================
// API types
// ============================================================
#[derive(Serialize)]
struct BoardResponse {
    board: Vec<Vec<Option<CellData>>>,
    side_to_move: u8,
    move_number: u32,
    black_pieces: usize,
    white_pieces: usize,
    game_result: Option<String>,
    mode: &'static str,
    move_log: Vec<String>,
    score_history: Vec<i32>,
    game_over: bool,
}

#[derive(Serialize)]
struct CellData {
    piece: String,
    color: u8,
    name: String,
    value: i32,
    is_royal: bool,
}

#[derive(Serialize)]
struct MoveListResponse {
    moves: Vec<MoveData>,
    from: [usize; 2],
}

#[derive(Serialize)]
struct MoveData {
    to: [usize; 2],
    promotion: bool,
    is_igui: bool,
    captured: Option<String>,
}

#[derive(Deserialize)]
struct MoveBody {
    from: [usize; 2],
    to: [usize; 2],
    promotion: Option<bool>,
}

#[derive(Deserialize)]
struct AiMoveBody {
    depth: Option<u32>,
    time_limit: Option<u64>,
}

#[derive(Serialize)]
struct MoveResponse {
    ok: bool,
    error: Option<String>,
    board: Option<BoardResponse>,
    last_move: Option<LastMoveData>,
}

#[derive(Serialize)]
struct LastMoveData {
    from: [usize; 2],
    to: [usize; 2],
}

#[derive(Serialize)]
struct NewGameBody {
    ok: bool,
}

#[derive(Serialize)]
struct PieceInfoResponse {
    abbrev: String,
    name: String,
    value: i32,
    promotes_to: Option<String>,
    slide_directions: usize,
    jump_destinations: usize,
    has_hook: bool,
    area_steps: u8,
    has_range_capture: bool,
    has_igui: bool,
}

// ============================================================
// Helpers
// ============================================================
fn board_to_response(state: &GameState) -> BoardResponse {
    let mut board_data = Vec::with_capacity(36);
    for r in 0..36 {
        let mut row = Vec::with_capacity(36);
        for c in 0..36 {
            let cell = state.board.get(r, c);
            if let Some(piece) = cell {
                row.push(Some(CellData {
                    piece: piece.abbrev().to_string(),
                    color: if piece.color == Color::Black { 0 } else { 1 },
                    name: piece.name().to_string(),
                    value: piece.value(),
                    is_royal: piece.is_royal(),
                }));
            } else {
                row.push(None);
            }
        }
        board_data.push(row);
    }

    let result = state.board.game_result().map(|r| match r {
        GameResult::BlackWins => "black_wins".into(),
        GameResult::WhiteWins => "white_wins".into(),
        GameResult::Draw => "draw".into(),
    });

    BoardResponse {
        board: board_data,
        side_to_move: if state.board.side_to_move() == Color::Black { 0 } else { 1 },
        move_number: state.board.move_number(),
        black_pieces: state.board.piece_count(Color::Black),
        white_pieces: state.board.piece_count(Color::White),
        game_result: result,
        mode: "human_vs_ai",
        move_log: state.move_log.clone(),
        score_history: state.score_history.clone(),
        game_over: state.game_over,
    }
}

fn get_moves_for_square(state: &GameState, r: usize, c: usize) -> Vec<MoveData> {
    // Check if there's a piece of the side to move at this square
    let cell = state.board.get(r, c);
    match cell {
        Some(piece) if piece.color == state.board.side_to_move() => {},
        _ => return vec![],
    }

    let all_moves = state.board.legal_moves();
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for m in &all_moves {
        let from = m.from();
        if from.row == r && from.col == c {
            let to = m.to();
            let key = (to.row, to.col, m.is_promotion(), m.is_igui());
            if seen.insert(key) {
                result.push(MoveData {
                    to: [to.row, to.col],
                    promotion: m.is_promotion(),
                    is_igui: m.is_igui(),
                    captured: m.captured().map(|s| s.to_string()),
                });
            }
        }
    }
    result
}

fn apply_move_to_board(state: &mut GameState, fr: usize, fc: usize, tr: usize, tc: usize, promotion: bool) -> bool {
    let piece = match state.board.get(fr, fc) {
        Some(p) => p.abbrev().to_string(),
        None => return false,
    };

    let side = if state.board.side_to_move() == Color::Black { "Black" } else { "White" };

    let from_sq = taikyokushogi::BOARD_SIZE * fr + fc;
    let to_sq = taikyokushogi::BOARD_SIZE * tr + tc;

    let legal_moves = state.board.legal_moves();
    for m in &legal_moves {
        let mv = m;
        // Access raw move data
        let raw = mv.raw();
        if raw.from_sq as usize == from_sq && raw.to_sq as usize == to_sq && mv.is_promotion() == promotion {
            state.board.apply(mv);
            state.record_move(&side, &piece, fr, fc, tr, tc, promotion);
            if state.board.game_result().is_some() {
                state.game_over = true;
            }
            return true;
        }
    }
    false
}

// ============================================================
// HTTP Handlers
// ============================================================
async fn get_state(State(app): State<Arc<AppState>>) -> Json<BoardResponse> {
    let state = app.board.lock().unwrap();
    Json(board_to_response(&state))
}

async fn get_moves(
    State(app): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, usize>>,
) -> Json<MoveListResponse> {
    let r = params.get("r").copied().unwrap_or(0).min(35);
    let c = params.get("c").copied().unwrap_or(0).min(35);
    let state = app.board.lock().unwrap();
    let moves = get_moves_for_square(&state, r, c);
    Json(MoveListResponse { moves, from: [r, c] })
}

async fn get_piece_info(Path(abbrev): Path<String>) -> Json<PieceInfoResponse> {
    let info = taikyokushogi::piece_info(&abbrev);
    match info {
        Some(i) => Json(PieceInfoResponse {
            abbrev: i.abbrev.to_string(),
            name: i.name.to_string(),
            value: i.value,
            promotes_to: i.promotes_to.map(|s| s.to_string()),
            slide_directions: i.slide_directions,
            jump_destinations: i.jump_destinations,
            has_hook: i.has_hook,
            area_steps: i.area_steps,
            has_range_capture: i.has_range_capture,
            has_igui: i.has_igui,
        }),
        None => Json(PieceInfoResponse {
            abbrev: abbrev.clone(),
            name: abbrev,
            value: 0,
            promotes_to: None,
            slide_directions: 0,
            jump_destinations: 0,
            has_hook: false,
            area_steps: 0,
            has_range_capture: false,
            has_igui: false,
        }),
    }
}

async fn post_new_game(State(app): State<Arc<AppState>>) -> Json<NewGameBody> {
    let mut state = app.board.lock().unwrap();
    state.reset();
    Json(NewGameBody { ok: true })
}

async fn post_move(
    State(app): State<Arc<AppState>>,
    Json(body): Json<MoveBody>,
) -> Json<MoveResponse> {
    let mut state = app.board.lock().unwrap();
    if state.game_over {
        return Json(MoveResponse {
            ok: false,
            error: Some("Game is over".into()),
            board: None,
            last_move: None,
        });
    }

    let promotion = body.promotion.unwrap_or(false);
    let ok = apply_move_to_board(&mut state, body.from[0], body.from[1], body.to[0], body.to[1], promotion);

    if !ok {
        return Json(MoveResponse {
            ok: false,
            error: Some("Illegal move".into()),
            board: None,
            last_move: None,
        });
    }

    let resp = board_to_response(&state);
    Json(MoveResponse {
        ok: true,
        error: None,
        board: Some(resp),
        last_move: Some(LastMoveData {
            from: body.from,
            to: body.to,
        }),
    })
}

async fn post_ai_move(
    State(app): State<Arc<AppState>>,
    Json(body): Json<AiMoveBody>,
) -> Json<MoveResponse> {
    let depth = body.depth.unwrap_or(0);
    let time_limit = body.time_limit.unwrap_or(30000);

    let mut state = app.board.lock().unwrap();
    if state.game_over {
        return Json(MoveResponse {
            ok: false,
            error: Some("Game is over".into()),
            board: None,
            last_move: None,
        });
    }

    let (fr, fc, tr, tc, promotion) = if depth == 0 {
        // Random move
        match state.board.random_move() {
            Some(m) => {
                let from = m.from();
                let to = m.to();
                (from.row, from.col, to.row, to.col, m.is_promotion())
            }
            None => {
                state.game_over = true;
                return Json(MoveResponse {
                    ok: false,
                    error: Some("No legal moves".into()),
                    board: None,
                    last_move: None,
                });
            }
        }
    } else {
        // AI search
        let result = state.board.search(depth, time_limit);
        match result.best_move {
            Some(m) => {
                let from = m.from();
                let to = m.to();
                (from.row, from.col, to.row, to.col, m.is_promotion())
            }
            None => {
                state.game_over = true;
                return Json(MoveResponse {
                    ok: false,
                    error: Some("No legal moves".into()),
                    board: None,
                    last_move: None,
                });
            }
        }
    };

    let piece = match state.board.get(fr, fc) {
        Some(p) => p.abbrev().to_string(),
        None => "?".into(),
    };
    let side = if state.board.side_to_move() == Color::Black { "Black" } else { "White" };

    // Apply the move using apply_by_coord
    state.board.apply_by_coord(fr, fc, tr, tc, promotion);
    state.record_move(&side, &piece, fr, fc, tr, tc, promotion);
    if state.board.game_result().is_some() {
        state.game_over = true;
    }

    let resp = board_to_response(&state);
    Json(MoveResponse {
        ok: true,
        error: None,
        board: Some(resp),
        last_move: Some(LastMoveData {
            from: [fr, fc],
            to: [tr, tc],
        }),
    })
}

async fn post_undo(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut state = app.board.lock().unwrap();
    let ok = state.board.undo();
    if ok {
        state.move_log.pop();
        if state.half_move > 0 {
            state.half_move -= 1;
        }
        state.game_over = false;
        Json(serde_json::json!({"ok": true}))
    } else {
        Json(serde_json::json!({"ok": false, "error": "Nothing to undo"}))
    }
}

// ============================================================
// Static file serving for the web frontend
// ============================================================
async fn serve_frontend() -> impl IntoResponse {
    // Serve the index.html from the web/dist directory
    let content = tokio::fs::read_to_string("web/dist/index.html").await
        .unwrap_or_else(|_| "<html><body><h1>Frontend not built</h1><p>Run: cd web && npm install && npm run build</p></body></html>".to_string());
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/html; charset=utf-8"),
        )],
        content,
    )
}

async fn serve_assets(Path(path): Path<String>) -> impl IntoResponse {
    let file_path = format!("web/dist/{}", path);
    match tokio::fs::read(&file_path).await {
        Ok(data) => {
            let content_type = match path.rsplit('.').next().unwrap_or("") {
                "js" => "application/javascript",
                "css" => "text/css",
                "html" => "text/html",
                "png" => "image/png",
                "svg" => "image/svg+xml",
                "ico" => "image/x-icon",
                "woff2" => "font/woff2",
                _ => "application/octet-stream",
            };
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, HeaderValue::from_static(content_type))],
                data,
            )
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            [(header::CONTENT_TYPE, HeaderValue::from_static("text/plain"))],
            b"404 Not Found".to_vec(),
        ),
    }
}

// ============================================================
// Main
// ============================================================
#[tokio::main]
async fn main() {
    println!("Taikyoku Shogi Server starting on http://0.0.0.0:5173");

    let state = Arc::new(AppState {
        board: Mutex::new(GameState::new()),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        // API routes
        .route("/api/state", get(get_state))
        .route("/api/moves", get(get_moves))
        .route("/api/piece-info/{abbrev}", get(get_piece_info))
        .route("/api/new-game", post(post_new_game))
        .route("/api/move", post(post_move))
        .route("/api/ai-move", post(post_ai_move))
        .route("/api/undo", post(post_undo))
        // Frontend
        .route("/", get(serve_frontend))
        .route("/:path", get(serve_assets))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("Taikyoku Shogi Server starting on http://0.0.0.0:8000");
    axum::serve(listener, app).await.unwrap();
}