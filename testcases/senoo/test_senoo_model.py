import numpy as np
import pytest
from unsflow.senoo import SenooDiffuser, SenooConfig, SenooFlowField


def test_senoo_config_creation():
    """Test both dimensional and non-dimensional configuration creation."""
    # Dimensional
    cfg_dim = SenooConfig.from_dimensional(
        r1=0.25,
        r2=0.50,
        b1=0.014,
        alpha1_mean=17.0,
        V1_mean=45.0
    )
    assert cfg_dim.r1 == 0.25
    assert cfg_dim.r2 == 0.50
    assert cfg_dim.b1 == 0.014
    assert cfg_dim.b2 == 0.014 # Parallel wall default
    assert cfg_dim.alpha1_mean == 17.0

    # Non-dimensional
    cfg_nondim = SenooConfig.from_nondimensional(
        b_over_r0=0.0558,
        re_over_r0=2.0,
        Rb=4.2e4,
        alpha1_mean=17.0,
        distortion_ur=0.1
    )
    assert np.isclose(cfg_nondim.r2 / cfg_nondim.r1, 2.0)
    assert np.isclose(cfg_nondim.b1 / cfg_nondim.r1, 0.0558)
    assert cfg_nondim.distortion_ur == 0.1


def test_senoo_unstalled_flow():
    """Test solver on regular unstalled operating point (Senoo et al. 1977, alpha=17 deg)."""
    cfg = SenooConfig.from_dimensional(
        r1=0.251,
        r2=0.502,
        b1=0.014,
        alpha1_mean=17.0,
        V1_mean=45.0,
        nr=60,
        nspan=30
    )
    diffuser = SenooDiffuser(cfg)
    diffuser.print_info()

    field = diffuser.solve()
    assert isinstance(field, SenooFlowField)

    # Coordinates
    assert len(field.r) == 60
    assert len(field.span) == 30
    assert field.vr.shape == (60, 30)
    assert field.vt.shape == (60, 30)
    assert field.V.shape == (60, 30)
    assert field.alpha.shape == (60, 30)

    # Static pressure must increase along radius in a diffuser
    assert field.p[-1] > field.p[0]
    assert field.Cp[-1] > 0.0

    # Flow should be unstalled
    assert not field.is_stalled

    # Angular momentum: r * Ut should be roughly conserved in the core
    r_core_inlet = field.r[0] * field.Ut_hub[0]
    r_core_outlet = field.r[-1] * field.Ut_hub[-1]
    # Minor reduction due to wall skin friction
    assert r_core_outlet < r_core_inlet
    assert r_core_outlet > 0.6 * r_core_inlet

    field.print_summary()


def test_senoo_reverse_flow_stall():
    """Test solver at low flow angle (alpha=8 deg) where reverse flow occurs."""
    cfg = SenooConfig.from_dimensional(
        r1=0.251,
        r2=0.502,
        b1=0.014,
        alpha1_mean=8.0,
        V1_mean=45.0,
        nr=60,
        nspan=30
    )
    diffuser = SenooDiffuser(cfg)
    field = diffuser.solve()

    # Flow must exhibit reverse flow / stall
    assert field.is_stalled
    assert field.r_stall_hub is not None or field.r_stall_shroud is not None


def test_critical_flow_angle_search():
    """Test bisection search for critical inlet flow angle."""
    cfg = SenooConfig.from_dimensional(
        r1=0.251,
        r2=0.502,
        b1=0.014,
        alpha1_mean=15.0,
        V1_mean=45.0
    )
    diffuser = SenooDiffuser(cfg)
    alpha_cr = diffuser.find_critical_flow_angle(tol=0.2, max_iter=15)
    print(f"Computed critical flow angle: {alpha_cr:.2f} deg")

    # In Senoo's papers for b/r0 ~ 0.055, critical angle is around 9 - 13 deg
    assert 7.0 <= alpha_cr <= 16.0


def test_pinched_diffuser_geometry():
    """Test pinched / tapered diffuser geometry where b decreases along radius."""
    cfg = SenooConfig.from_dimensional(
        r1=0.25,
        r2=0.50,
        b1=0.015,
        b2=0.010, # pinched walls
        alpha1_mean=16.0,
        V1_mean=45.0,
        nr=40,
        nspan=20
    )
    diffuser = SenooDiffuser(cfg)
    field = diffuser.solve()

    assert field.b_r[0] == 0.015
    assert field.b_r[-1] == 0.010
    assert field.vr.shape == (40, 20)


def test_compressible_regime():
    """Test compressible formulation with Mach number ~ 0.6."""
    cfg = SenooConfig.from_dimensional(
        r1=0.25,
        r2=0.50,
        b1=0.014,
        alpha1_mean=18.0,
        V1_mean=200.0,
        mach1=0.6,
        compressible=True,
        nr=40,
        nspan=20
    )
    diffuser = SenooDiffuser(cfg)
    field = diffuser.solve()

    assert field.mach is not None
    assert field.mach.shape == (40, 20)
    assert np.all(field.rho > 0.0)
    assert field.p[-1] > field.p[0]


def test_to_dict_export():
    """Test conversion to dictionary."""
    cfg = SenooConfig.from_dimensional(
        r1=0.25,
        r2=0.40,
        b1=0.014,
        alpha1_mean=17.0,
        nr=20,
        nspan=10
    )
    diffuser = SenooDiffuser(cfg)
    field = diffuser.solve()
    d = field.to_dict()

    assert 'r' in d
    assert 'vr' in d
    assert 'vt' in d
    assert 'alpha' in d
    assert 'p' in d
    assert d['vr'].shape == (20, 10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
