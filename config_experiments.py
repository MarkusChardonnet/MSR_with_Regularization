
"""
parameters = {
    "problem": ["rank1", "rank2", "rank5", "rank2_kernel5", "rank5_kernel5", "2d_rot4", "2d_rot8", "2d_rot8_flip"],
    "lam_reg": [0.001, 0.005],
    "epochs": [100, 1000],
    "model": ["share_fc"], # "share_conv"],
    "ntasks": [50, 100, 1000],
    "trainer": ["base", "sparsity"]
}


parameters = {
    "problem": ["rank1"],
    "lam_reg": [0.005],
    "epochs": [100],
    "model": ["share_fc"],
    "ntasks": [50],
    "trainer": ["base", "sparsity"]
}

parameters = {
    "problem": ["rank1" "rank2", "rank5"],
    "lam_reg": [0.001, 0.005],
    "epochs": [100, 1000],
    "model": ["share_fc"],
    "ntasks": [50, 100, 1000],
    "trainer": ["base", "sparsity"]
}
"""

parameters = {
    "problem": ["rank2", "rank5", "rank2_kernel5", "rank5_kernel5"],
    "lam_reg": [0.001, 0.005],
    "epochs": [100, 1000],
    "model": ["share_fc"],
    "ntasks": [20, 50, 100, 1000],
    "trainer": ["base", "sparsity"]
}