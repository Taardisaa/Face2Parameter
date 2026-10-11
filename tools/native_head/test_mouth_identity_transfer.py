"""Full-detail control retention and intact native oral-interface contracts."""
import unittest
import numpy as np
from tools.native_head.mouth_identity_transfer import full_control_field,deform_oral
from tools.native_head.mouth_contact_repair import front_exposed
from tools.native_head.mother_identity import build

class MouthContracts(unittest.TestCase):
    def test_nonaffine_detail_retained_at_every_source_node(self):
        controls=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.3,.2,.4],[.7,.1,.2]],float)
        values=np.stack([np.sin(controls*2),controls**2+controls[:,[1,2,0]]],axis=-1)
        field,receipt=full_control_field(controls,controls,values)
        np.testing.assert_allclose(field,values,atol=1e-12,rtol=0)
        beta=np.array([.3,-.7])
        np.testing.assert_allclose(field@beta,values@beta,atol=1e-12,rtol=0)
        self.assertFalse(receipt['source_control_values_filtered'])

    def test_duplicate_layers_are_rejected_without_averaging(self):
        controls=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0]],float)
        values=np.ones((5,3,2));values[-1]=2
        with self.assertRaisesRegex(ValueError,'Duplicate control'):
            full_control_field(controls,controls,values)

    def test_oral_recompute_preserves_literal_interface_and_exterior(self):
        source=np.array([[0,0,0],[1,0,0],[0,1,0],[.3,.3,-.2],[4,4,4]],float)
        faces=np.array([[0,3,1],[1,3,2],[2,3,0]])
        target=source.copy();target[:3]=source[:3]*1.2+[.2,.1,-.3]
        result,report=deform_oral(source,target,faces,np.arange(3),np.arange(3),np.array([3]))
        np.testing.assert_array_equal(result[:3],target[:3])
        np.testing.assert_array_equal(result[4],source[4])
        np.testing.assert_allclose(result[3],source[3]*1.2+[.2,.1,-.3],atol=1e-12)
        self.assertTrue(report['converged'])

    def test_front_pin_ray_distinguishes_actual_hidden_layer(self):
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,-1],[.2,.2,1]],float)
        faces=np.array([[0,1,2]])
        np.testing.assert_array_equal(front_exposed(vertices,faces,[3,4]),[4])

    def test_generic_path_cannot_enable_affine_preview(self):
        with self.assertRaisesRegex(ValueError,'cannot use affine'):
            build(None,None,None,0,None,mouth_detail_preview=True,generic_mouth=True)

if __name__=='__main__':unittest.main()
