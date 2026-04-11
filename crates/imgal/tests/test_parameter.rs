use imgal::parameter::{abbe_diffraction_limit, omega};

/// Tests that `abbe_diffraction_limit` returns the expected result for a given
/// wavelength and numerical aperature.
#[test]
fn parameter_abbe_diffraction_limit_expected_results() {
    assert_eq!(abbe_diffraction_limit(465, 0.40), 581.25);
    assert_eq!(abbe_diffraction_limit(465, 0.75), 310.0);
    assert_eq!(abbe_diffraction_limit(465, 1.45), 160.3448275862069);
    assert_eq!(abbe_diffraction_limit(570, 0.40), 712.5);
    assert_eq!(abbe_diffraction_limit(570, 0.75), 380.0);
    assert_eq!(abbe_diffraction_limit(570, 1.45), 196.55172413793105);
}

/// Tests that `omega` returns the expected result for a given period.
#[test]
fn parameter_omega_expected_results() {
    assert_eq!(omega(10.0), 0.6283185307179586);
    assert_eq!(omega(12.5), 0.5026548245743669);
    assert_eq!(omega(15.0), 0.41887902047863906);
}
