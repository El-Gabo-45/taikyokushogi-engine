use rusqlite::Connection;
fn main() {
    let conn = Connection::open("training_data/games.db").unwrap();
    let mut stmt = conn.prepare("SELECT id, result, total_moves, depth, duration_ms FROM games").unwrap();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, i64>(4)?))
    }).unwrap();
    for r in rows {
        let (id, result, moves, depth, dur) = r.unwrap();
        println!("Game {}: result={} moves={} depth={} ms={}", id, result, moves, depth, dur);
    }
    let mut stmt2 = conn.prepare("SELECT game_id, move_number, from_sq, to_sq, promo, captured, side FROM moves LIMIT 10").unwrap();
    let rows2 = stmt2.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, i64>(2)?, row.get::<_, i64>(3)?, row.get::<_, i64>(4)?, row.get::<_, i64>(5)?, row.get::<_, i64>(6)?))
    }).unwrap();
    for r in rows2 {
        let (gid, mn, from, to, promo, cap, side) = r.unwrap();
        println!("  Move: game={} #{} from={} to={} promo={} cap={} side={}", gid, mn, from, to, promo, cap, side);
    }
}
