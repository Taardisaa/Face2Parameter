import unittest
from tools.character_authoring.capture_bridge_il_audit import normalized, types


class PreservationTests(unittest.TestCase):
    def test_pe_placement_is_not_code_change(self):
        a = '// Method begins at RVA 0x123\n.field x at I_000123\n.data cil I_000123 = bytearray (01 02)'
        b = '// Method begins at RVA 0x456\n.field x at I_000456\n.data cil I_000456 = bytearray (01 02)'
        self.assertEqual(normalized(a), normalized(b))

    def test_instruction_and_data_changes_remain_visible(self):
        self.assertNotEqual(normalized('IL_0000: ldc.i4 0x123'), normalized('IL_0000: ldc.i4 0x456'))
        self.assertNotEqual(normalized('.data cil I_000123 = bytearray (01 02)'),
                            normalized('.data cil I_000456 = bytearray (01 03)'))

    def test_empty_and_incomplete_text_cannot_pass(self):
        for text in ('', '// no type', '.class private A\n{'):
            with self.assertRaises(ValueError):
                types(text)

    def test_nested_type_included_in_parent_comparison(self):
        a = '.class private A\n{\n\t.class private B\n\t{\n\t ldstr "original"\n\t} // end of class A/B\n} // end of class A\n'
        b = a.replace('original', 'changed')
        self.assertEqual(set(types(a)), {'A'})
        self.assertNotEqual(types(a)['A'], types(b)['A'])


if __name__ == '__main__':
    unittest.main()
