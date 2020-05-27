import torch
import ssdn
import logging
import os

from torch.utils.data import Dataset
from ssdn.datasets import NoisyDataset
from ssdn.denoiser import Denoiser
from ssdn.train import DenoiserTrainer
from ssdn.params import PipelineOutput
from ssdn.cfg import DEFAULT_RUN_DIR
from typing import Callable, Dict
from tqdm import tqdm


logger = logging.getLogger("ssdn.eval")


class DenoiserEvaluator(DenoiserTrainer):
    """Class to start evaluation of a dataset on a trained Denoiser model.
    Initialise with constructor and set evaluation dataset with `set_test_data()`
    before calling `evaluate()`.

    Args:
        target_path (str): Path to weights or training file to evaluate.
        runs_dir (str, optional): Root directory to create run directory.
            Defaults to DEFAULT_RUN_DIR.
        run_dir (str, optional): Explicit run directory name, will automatically
            generate using configuration if not provided. Defaults to None.
    """

    def __init__(
        self, target_path: str, runs_dir: str = DEFAULT_RUN_DIR, run_dir: str = None,
    ):
        super().__init__({})
        state_dict = torch.load(target_path, map_location="cpu")
        if "denoiser" in state_dict:
            self.load_state_dict(state_dict)
        else:
            self.denoiser = Denoiser.from_state_dict(state_dict)
        self.cfg = self.denoiser.cfg
        self._run_dir = run_dir
        self.init_state()

    def evaluate(self):
        self.reset_metrics(train=False)
        if self.denoiser is None:
            raise RuntimeError("Denoiser not initialised for evaluation")
        # Ensure writer is initialised
        _ = self.writer
        ssdn.logging_helper.setup(self.run_dir_path, "log.txt")
        logger.info(ssdn.utils.separator())
        logger.info("Loading Test Dataset...")
        self.testloader, self.testset, self.test_sampler = self.test_data()
        logger.info("Loaded Test Dataset.")

        logger.info(ssdn.utils.separator())
        logger.info("EVALUATION STARTED")
        logger.info(ssdn.utils.separator())

        dataloader = tqdm(self.testloader)
        save_callback = self.evaluation_output_callback(self.testset)
        self._evaluate(dataloader, save_callback)
        logger.info(self.eval_state_str("EVALUATION RESULT"))
        logger.info(ssdn.utils.separator())
        logger.info("EVALUATION FINISHED")
        logger.info(ssdn.utils.separator())

    @property
    def run_dir(self) -> str:
        """The run path to use for this run. When this method is first called
        a new directory name will be generated using the next run ID and current
        configuration.

        Returns:
            str: Run directory name, note this is not a full path.
        """
        if self._run_dir is None:
            config_name = self.config_name()
            next_run_id = self.next_run_id()
            run_dir_name = "{:05d}-eval-{}".format(next_run_id, config_name)
            self._run_dir = run_dir_name

        return self._run_dir

    def evaluation_output_callback(
        self, dataset: Dataset
    ) -> Callable[[int, Dict], None]:
        """Callback that saves all dataset images for evaluation with an associated
        PSNR record.

        Args:
            dataset (Dataset): Dataset which determines how many images are saved in case
                of repeats.

        Returns:
            Callable[[int, Dict], None]: Callback function for evaluator.
        """

        def callback(output_0_index: int, outputs: Dict):
            remaining = (len(dataset) - 1) - output_0_index
            inp = outputs[PipelineOutput.INPUTS][NoisyDataset.INPUT]
            batch_size = inp.shape[0]
            if remaining > 0:
                bis = range(min(remaining, batch_size))
                output_dir = os.path.join(self.run_dir_path, "eval_imgs")
                os.makedirs(output_dir, exist_ok=True)
                fileformat = "img_{index:05}_{desc}.png"
                self.save_image_outputs(
                    outputs, output_dir, fileformat, batch_indexes=bis
                )
                with open(os.path.join(self.run_dir_path, "psnrs.csv"), "a") as f:
                    if output_0_index == 0:
                        fields = ["id"] + list(self.img_outputs(prefix="psnr").values())
                        f.write(",".join(fields) + "\n")
                    # FIXME: Doing a second PSNR calculation
                    values = []
                    for key in self.img_outputs(prefix="psnr"):
                        values += [self.calculate_psnr(outputs, key, unpad=True)]
                    for i in range(batch_size):
                        str_lst = ["{:04d}".format(output_0_index + i)]
                        str_lst += ["{:.4f}".format(value[i]) for value in values]
                        f.write(",".join(str_lst) + "\n")

        return callback
