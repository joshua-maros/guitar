pub const GUITAR_STRING_POINTS: usize = 1024;
pub const GUITAR_STRING_LENGTH: f64 = 0.65;
pub const GUITAR_STRING_DENSITY: f64 = 5.25e-3;
pub const GUITAR_STRING_TENSION: f64 = 120.0;
pub const INTERNAL_DISSIPATIVE_TERM_OF_VISCOELASTIC_TYPE: f64 = 9e-8;
pub const DELTA_TIME: f64 = 1e-7;
pub const MAX_SIMULATION_TIME: f64 = 3.0;
pub const SCREEN_REFRESH_INTERVAL: usize = 5_000;

/// The board is modeled as a BOARD_RES x BOARD_RES grid of nodes.
pub const BOARD_RES: usize = 20;
pub const BOARD_PARAMS: usize = BOARD_RES * BOARD_RES * 3;
pub const BOARD_SIZE: f64 = 0.5;
pub const BOARD_EL_SIZE: f64 = BOARD_SIZE / (BOARD_RES as f64 - 1.0);
pub const BOARD_DISSIPATIVE_TERM: f64 = 0.005e-1;
pub const BOARD_DENSITY: f64 = 400.0;
pub const BOARD_ALPHA: f64 = 0.0029;
pub const BOARD_RIGIDITY: f64 = 200.0e6;
