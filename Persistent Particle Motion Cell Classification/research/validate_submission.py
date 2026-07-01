import numpy as np, pandas as pd, sys, os
# Strict checker for the Persistent Particle submission contract.
def check(sub_path='working/submission.csv', sample_path='dataset/sample_submission.csv'):
    assert os.path.exists(sub_path), f'missing {sub_path}'
    sub=pd.read_csv(sub_path); samp=pd.read_csv(sample_path)
    assert list(sub.columns)==['sample_id','motion_class'], ('cols',sub.columns.tolist())
    assert len(sub)==len(samp), ('len',len(sub),len(samp))
    assert sub['sample_id'].is_unique, 'dup ids'
    assert set(sub['sample_id'])==set(samp['sample_id']), 'id set mismatch'
    assert list(sub['sample_id'])==list(samp['sample_id']), 'id ORDER mismatch vs sample_submission'
    mc=sub['motion_class']
    assert mc.notna().all(), 'NaN present'
    assert np.isfinite(mc.to_numpy()).all(), 'non-finite'
    assert (mc==mc.astype(int)).all(), 'non-integer'
    assert mc.min()>=0 and mc.max()<=19, ('range',mc.min(),mc.max())
    print('SUBMISSION VALID:', sub.shape, 'class range',int(mc.min()),int(mc.max()))
    print('pred class dist:', np.bincount(mc.astype(int),minlength=20))
    return True
if __name__=='__main__':
    check(*(sys.argv[1:] if len(sys.argv)>1 else []))
