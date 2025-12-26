#![doc(
    html_logo_url = "https://raw.githubusercontent.com/imgal-sc/imgal/refs/heads/main/docs/assets/png/imgal_logo.png"
)]
//! The `imgal` (**IM**a**G**e **A**lgorithm **L**ibrary) crate is a fast and
//! open-source scientific image processing and algorithm library. This library
//! is directly inspired by [imagej-ops](https://github.com/imagej/imagej-ops/),
//! [SciJava Ops](https://github.com/scijava/scijava),
//! [ImgLib2](https://github.com/imglib/imglib2), and the ImageJ2 ecosystem.
//! `imgal` library aims to offer users access to fast and well documented
//! scientific image processing and analysis algorithms.
//!
//! ## Crate Status
//!
//! This crate is still under active development and it's API is not stable.
pub mod colocalization;
pub mod distribution;
pub mod error;
pub mod filter;
pub mod image;
pub mod integration;
pub mod kernel;
pub mod overlay;
pub mod parameter;
pub mod phasor;
pub mod simulation;
pub mod spatial;
pub mod statistics;
pub mod threshold;
pub mod traits;
pub mod transform;
