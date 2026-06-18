import os
import sys
import torch
import unittest
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.configs import BaseLoopConfig, PreludeCodaConfig
from src.models.base_loop import SimpleLoopModel
from src.models.prelude_coda import PreludeCodaLoopModel
from src.models.trm import TRMStrippedModel

class TestArchitectures(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 111
        self.bsz = 2
        self.seq_len = 10
        self.max_train_loops = 6
        
        self.inputs = {
            "input_ids": torch.randint(0, self.vocab_size, (self.bsz, self.seq_len)),
            "hops": torch.tensor([3, 4])
        }

    def test_base_loop_model(self):
        config_path = os.path.join(project_root, "config", "arch", "base_loop.yaml")
        config = BaseLoopConfig.from_yaml(config_path, vocab_size=self.vocab_size)
        model = SimpleLoopModel(config)
        
        out = model(self.inputs)
        
        self.assertTrue(hasattr(out, 'logits'))
        self.assertTrue(hasattr(out, 'predictions'))
        self.assertTrue(hasattr(out, 'last_hidden_states_loops'))
        self.assertTrue(hasattr(out, 'hops'))
        
        expected_history_shape = (self.bsz, config.max_train_loops, self.seq_len, config.d_model)
        self.assertEqual(out.last_hidden_states_loops.shape, expected_history_shape)
        
        expected_logits_shape = (self.bsz, self.seq_len, self.vocab_size)
        self.assertEqual(out.logits.shape, expected_logits_shape)
        
        expected_predictions_shape = (self.bsz,)
        self.assertEqual(out.predictions.shape, expected_predictions_shape)

    def test_prelude_coda_model(self):
        config_path = os.path.join(project_root, "config", "arch", "prelude_coda.yaml")
        config = PreludeCodaConfig.from_yaml(config_path, vocab_size=self.vocab_size)
        model = PreludeCodaLoopModel(config)
        
        out = model(self.inputs)
        
        expected_history_shape = (self.bsz, config.max_train_loops, self.seq_len, config.d_model)
        self.assertEqual(out.last_hidden_states_loops.shape, expected_history_shape)
        self.assertEqual(out.predictions.shape, (self.bsz,))

    def test_prelude_coda_mismatched(self):
        config_path = os.path.join(project_root, "config", "arch", "prelude_coda_mismatch.yaml")
        config = PreludeCodaConfig.from_yaml(config_path, vocab_size=self.vocab_size)
        model = PreludeCodaLoopModel(config)
        
        out = model(self.inputs)
        
        expected_history_shape = (self.bsz, config.max_train_loops, self.seq_len, config.d_model)
        self.assertEqual(out.last_hidden_states_loops.shape, expected_history_shape)
        self.assertEqual(out.predictions.shape, (self.bsz,))

    def test_trm_stripped_model(self):
        config_path = os.path.join(project_root, "config", "arch", "trm.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
            
        model = TRMStrippedModel(config=raw_config, vocab_size=self.vocab_size, max_train_loops=self.max_train_loops)
        
        out = model(self.inputs)
        
        # d_model for TRM is hidden_size
        expected_history_shape = (self.bsz, self.max_train_loops, self.seq_len, raw_config["hidden_size"])
        self.assertEqual(out.last_hidden_states_loops.shape, expected_history_shape)
        self.assertEqual(out.predictions.shape, (self.bsz,))

if __name__ == '__main__':
    import traceback
    
    log_path = os.path.join(os.path.dirname(__file__), "test_architectures_result.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("--- Architectures Test Log ---\n")
        
        tester = TestArchitectures()
        tester.setUp()
        
        try:
            f.write("1. Testing BaseLoopModel (BaseLoopConfig)...\n")
            tester.test_base_loop_model()
            f.write("[OK] BaseLoopModel passed. Output shape matched [bsz, max_loops, seq_len, d_model]\n\n")
            
            f.write("2. Testing PreludeCodaLoopModel Uniform (PreludeCodaConfig)...\n")
            tester.test_prelude_coda_model()
            f.write("[OK] PreludeCodaLoopModel Uniform passed. Output shape matched.\n\n")
            
            f.write("3. Testing PreludeCodaLoopModel Mismatched (d_loop != d_model)...\n")
            tester.test_prelude_coda_mismatched()
            f.write("[OK] PreludeCodaLoopModel Mismatched passed. Output shape matched and projections worked.\n\n")
            
            f.write("4. Testing TRMStrippedModel (TRM module independent)...\n")
            tester.test_trm_stripped_model()
            f.write("[OK] TRMStrippedModel passed. Output shape matched.\n\n")
            
            f.write("SUCCESS: All architecture tests passed perfectly!\n")
            print(f"All tests passed! Logs saved to {log_path}")
        except Exception as e:
            f.write("\nFAILED: Error occurred during testing:\n")
            f.write(traceback.format_exc())
            print(f"Tests failed! Please check {log_path} for details.")
