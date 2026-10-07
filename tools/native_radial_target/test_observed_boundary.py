import copy
import unittest
import numpy as np
from tools.abmx_replay.model import replay_apply,CACHE_FIELDS
from tools.native_radial_target.observed_boundary import prove_anchor,prove_observed_gap,exact_cache


class AnchorPolicyTests(unittest.TestCase):
    def source(self):
        cache={'_hasBaseline':True,'_lenBaseline':.3,'_sclBaseline':[1.2,.8,1.],
            '_posBaseline':[.2,-.1,.3],'_rotBaseline':[0.,0.,0.,1.],'_positionBaseline':[.2,-.1,.3],
            '_lenModForceUpdate':False,'_lenModNeedsPositionRestore':False,'_changedScale':True,
            '_changedRotation':True,'_changedPosition':True,'_forceApply':True}
        physical={'scale':[1.1,.9,1.],'length':1.,'position':[.01,.02,-.03],'rotation':[1.,-2.,3.]}
        return cache,physical

    def test_arbitrary_finite_incoming_invariance(self):
        cache,physical=self.source();anchor=prove_anchor(cache,physical);rng=np.random.default_rng(4829)
        states=[{'local_position':[0,0,0],'local_rotation_xyzw':[0,0,0,0],'local_scale':[0,0,0]},
                {'local_position':[1e30,-1e30,1e30],'local_rotation_xyzw':[1e30]*4,'local_scale':[-1e30,0,1e30]}]
        for _ in range(64):states.append({'local_position':rng.normal(size=3).tolist(),'local_rotation_xyzw':rng.normal(size=4).tolist(),'local_scale':rng.normal(size=3).tolist()})
        for before in states:
            result=replay_apply(before=before,cache=cache,coordinate_modifiers=[physical],coordinate=0,
                additional_modifiers=[],bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
            for k in result['after']:np.testing.assert_array_equal(result['after'][k],anchor['predicted_after_from_cache_not_incoming'][k])

    def test_unsupported_channels_and_length_refused(self):
        cache,physical=self.source()
        for key,value in (('length',1.1),('scale',[1,1,1]),('rotation',[0,0,0]),('position',[0,0,0])):
            p=copy.deepcopy(physical);p[key]=value
            with self.assertRaises(ValueError):prove_anchor(cache,p)
        cache['_lenModForceUpdate']=True
        with self.assertRaises(ValueError):prove_anchor(cache,physical)
        cache['_positionBaseline']=[0,0,0]
        self.assertTrue(prove_anchor(cache,physical)['all_finite_incoming_trs_independent_by_branch_structure'])

    def test_exact_all12_private_fields_not_tolerance(self):
        cache,_=self.source();self.assertEqual(len(CACHE_FIELDS),12)
        for key in CACHE_FIELDS:
            other=copy.deepcopy(cache)
            if isinstance(other[key],bool):other[key]=not other[key]
            elif isinstance(other[key],list):other[key][0]=float(np.nextafter(np.float32(other[key][0]),np.float32(np.inf)))
            else:other[key]=float(np.nextafter(np.float32(other[key]),np.float32(np.inf)))
            self.assertFalse(exact_cache(cache,other),key)

    def test_gap_proof_identity_cache_and_exclusions_refused(self):
        cache,p=self.source();after=prove_anchor(cache,p)['predicted_after_from_cache_not_incoming']
        event={'sequence':2,'frame':2,'completed_frame':2,'bone_name':'test','modifier_instance_identity':33,
            'additional_modifiers':[],'no_rotation_excluded':False,'is_during_h_scene':False,'coordinate_specific':False,
            'resolved_modifier':p,'before':{'bone_transform_id':44,**after,'cache':{'fields':cache}},
            'after':{'bone_transform_id':44,**after,'cache':{'fields':cache}}}
        previous=copy.deepcopy(event);previous['sequence']=1;previous['frame']=previous['completed_frame']=1
        self.assertTrue(prove_observed_gap(previous,event,p)['all12_private_fields_exactly_unchanged'])
        for mutate in ('cache','modifier','bone','exclusion','additional'):
            e=copy.deepcopy(event)
            if mutate=='cache':e['before']['cache']['fields']['_forceApply']=False
            if mutate=='modifier':e['modifier_instance_identity']+=1
            if mutate=='bone':e['before']['bone_transform_id']+=1
            if mutate=='exclusion':e['no_rotation_excluded']=True
            if mutate=='additional':e['additional_modifiers']=[p]
            with self.assertRaises(ValueError):prove_observed_gap(previous,e,p)


if __name__=='__main__':unittest.main()
