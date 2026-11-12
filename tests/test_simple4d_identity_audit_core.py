import numpy as np
import pytest
from cardioresp4d.experiments.simple4d_resp_v1_identity_audit import global_latent_pca, image_temporal_pca, correlation, align_surrogate, mutual_nearest_pairs, interpolate_neighbor_codes


def test_latent_pca_location_centering():
    t = np.linspace(0, 2*np.pi, 32, endpoint=False)
    signal = np.stack((np.sin(t), .1*np.cos(t), np.zeros_like(t)), axis=1)
    result = global_latent_pca([signal + [10,2,3], signal + [-3,5,1]])
    assert all(np.allclose(x.mean(0),0,atol=1e-14) for x in result['centered'])
    assert result['explained_variance_ratios'][0] > .98 and result['rank95'] == 1
    assert np.allclose(result['basis'], global_latent_pca([signal + [10,2,3], signal + [-3,5,1]])['basis'])


def test_image_surrogate_and_sign():
    t = np.sin(np.arange(20)*.3)
    images = 4 + t[:,None,None] * np.arange(12).reshape(3,4)
    out = image_temporal_pca(images)
    assert abs(correlation(t, out['scores'])['pearson']) > .999
    assert image_temporal_pca(np.ones((4,3,4)))['scores'] is None
    neg = -t
    assert correlation(t,neg)['pearson'] == pytest.approx(-1)
    assert np.allclose(align_surrogate(t,neg),t) and np.allclose(neg,-t)


def test_mutual_nearest_threshold():
    assert mutual_nearest_pairs([0,1,2], [.1,1.1,2.8], .25) == [(0,0,.1),(1,1,pytest.approx(.1))]
    assert len(mutual_nearest_pairs([0,1,2], [.1,1.1,2.8], 1)) == 3
    assert mutual_nearest_pairs([], [1], 1) == []


def test_neighbor_interpolation():
    previous = np.array([0.,2.]); following = np.array([4.,6.])
    result = interpolate_neighbor_codes(previous,following,0,.25,1)
    assert result['alpha'] == .25 and np.allclose(result['time'],[1,3]) and np.allclose(result['simple'],[2,4])
    with pytest.raises(ValueError): interpolate_neighbor_codes(previous,following,0,1,1)


def test_withheld_never_uses_target_code():
    def v2(target_code):
        return interpolate_neighbor_codes(np.array([1.,2.]),np.array([5.,6.]),0.,.25,1.)['time']
    assert np.array_equal(v2(np.ones(2)),v2(np.ones(2)*10000))
