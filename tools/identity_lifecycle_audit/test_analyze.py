from copy import deepcopy
import unittest
import numpy as np
from analyze import collect, reset, native_partial, F


def state():
    return {'_hasBaseline':True,'_lenBaseline':float(np.sqrt(10)),
            '_sclBaseline':[1,1,1],'_posBaseline':[0,4,3],'_rotBaseline':[0,0,0,1],
            '_positionBaseline':[0,1,3], '_lenModForceUpdate':False,'_lenModNeedsPositionRestore':False,
            '_changedScale':False,'_changedRotation':False,'_changedPosition':False,'_forceApply':False}


class LifecycleTests(unittest.TestCase):
    def test_collect_keeps_old_nonzero_history(self):
        s=state();local={'local_position':[0,7,8],'local_scale':[2,3,4],'local_rotation_xyzw':[0,0,0,1]}
        result=collect(s,local)
        self.assertFalse(result['length_history_recollected'])
        self.assertEqual(result['cache']['_positionBaseline'],s['_positionBaseline'])
        self.assertAlmostEqual(result['cache']['_lenBaseline'],s['_lenBaseline'],places=6)
        self.assertEqual(result['cache']['_posBaseline'],local['local_position'])

    def test_approximately_zero_history_recollects(self):
        s=state();s['_positionBaseline']=[1e-6,0,0]
        r=collect(s,{'local_position':[0,0,5],'local_scale':[1,1,1],'local_rotation_xyzw':[0,0,0,1]})
        self.assertTrue(r['length_history_recollected']);self.assertEqual(r['cache']['_lenBaseline'],5)

    def test_identity_reset_still_changes_position(self):
        s=state();r=reset(s)
        self.assertEqual(r['branch'],'latest_position_normalized_to_first_radius')
        self.assertNotEqual(r['local']['local_position'],s['_posBaseline'])
        self.assertAlmostEqual(np.linalg.norm(r['local']['local_position']),np.sqrt(10),places=6)
        self.assertFalse(r['cache']['_forceApply'])

    def test_forced_history_restore_and_missing_baseline(self):
        s=state();s['_lenModNeedsPositionRestore']=True
        r=reset(s);self.assertEqual(r['local']['local_position'],s['_positionBaseline'])
        self.assertFalse(r['cache']['_lenModNeedsPositionRestore'])
        s['_hasBaseline']=False;self.assertIsNone(reset(s)['local'])

    def test_two_reset_cleanup_causes_unowned_component_drift(self):
        s=state();first=reset(s)
        native={'local_position':[0,1,3],'local_scale':[1,1,1],'local_rotation_xyzw':[0,0,0,1]}
        mask={'local_position':[1],'local_scale':[],'local_rotation_xyzw':[]}
        partial=native_partial(first['local'],native,mask)
        second=reset(collect(s,partial)['cache'])
        later=native_partial(second['local'],{'local_position':[0,4,3]},mask)
        self.assertNotAlmostEqual(later['local_position'][2],3,places=5)
        self.assertEqual(later['local_position'][1],4)
        # Reversing remove/native alone cannot restore an unowned Z component.
        reversed_order=native_partial(first['local'],native,mask)
        self.assertNotAlmostEqual(reversed_order['local_position'][2],3,places=5)


if __name__=='__main__':unittest.main()
