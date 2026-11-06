"""Local unit tests. These assert delivered behavior, not chemical efficacy."""
import unittest
import torch
from common import ROOT,MoleculeTokenizer,read_rows,prepared_data
from model import TransformerNetModel2,LossModelAdapter
from runtime import make_diffusion,seed_all
from evaluate import score_rows


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_vocab_and_all_rows(self):
        tok=MoleculeTokenizer()
        self.assertEqual(len(tok),323)
        self.assertEqual(tok.token_to_id['[NH3+1]'],77)
        self.assertEqual(tok.tokens[72],tok.tokens[77])
        manifest=prepared_data()
        self.assertEqual(manifest['training_rows_used'],22169)
        for split in ('train_ChemBI','test_ChemBI'):
            for row in read_rows(ROOT/'data'/(split+'.csv')):
                expected=tok.original.encode_one(row['selfies'])[0].tolist()
                self.assertEqual(tok.encode(row['selfies']),expected)

    def test_original_truncation_and_decode(self):
        tok=MoleculeTokenizer()
        ids=tok.encode('[C]'*300)
        self.assertEqual(len(ids),256)
        self.assertEqual(ids[-1],2)
        self.assertEqual(ids.count(tok.token_to_id['[C]']),254)
        c=tok.token_to_id['[C]']
        self.assertEqual(tok.decode([1,c,2,c,0]),'[C][C]')

    def test_original_mask_and_loss(self):
        seed_all(11)
        net=TransformerNetModel2(323,hidden_size=32,model_channels=16,num_hidden_layers=1,
            num_attention_heads=4,intermediate_size=64).eval()
        captured=[]
        handle=net.input_transformers.register_forward_pre_hook(lambda m,a,k:captured.append(k['encoder_attention_mask'].clone()),with_kwargs=True)
        desc=torch.randn(2,6,768)
        mask=torch.tensor([[1,1,0,0,0,0]]*2)
        net(torch.randn(2,5,32),torch.tensor([1,1999]),desc,mask)
        handle.remove()
        torch.testing.assert_close(captured[0],mask[:,None,None,:].expand(-1,4,-1,-1))
        ids=torch.randint(0,323,(2,5))
        loss=make_diffusion().training_losses(LossModelAdapter(net),(ids,desc,mask,ids),torch.tensor([0,1999]))['loss'].mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_paired_evaluation(self):
        refs=[dict(id='a',smiles='CCO'),dict(id='b',smiles='CC')]
        preds=[dict(id='a',reference_smiles='CCO',generated_selfies='[C][C][O]'),
               dict(id='b',reference_smiles='CC',generated_selfies='')]
        metrics,_=score_rows(preds,refs,refs)
        self.assertEqual(metrics['validity'],0.5)
        self.assertEqual(metrics['morgan_r2_2048_paired_valid_mean'],1)
        with self.assertRaises(ValueError):
            score_rows([dict(preds[0],reference_smiles='CC')],refs,refs)


if __name__=='__main__':
    unittest.main(verbosity=2)
