"""Wall-clock training diagnostics kept outside deterministic training state."""

from contextlib import contextmanager, ExitStack
import math
import time

import torch


class TimingRecorder:
    def __init__(self, device="cpu"):
        self.device = device
        self.statistics = {}
        self.scopes = []

    def _synchronize(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    @contextmanager
    def measure(self, name):
        key = ".".join([*self.scopes,name])
        self._synchronize()
        started = time.perf_counter()
        self.scopes.append(name)
        failed = True
        try:
            yield
            failed = False
        finally:
            self._synchronize()
            elapsed = time.perf_counter()-started
            self.scopes.pop()
            values = self.statistics.setdefault(key,{"calls":0,"failed_calls":0,"total_seconds":0.,
                                                       "first_seconds":elapsed,"min_seconds":elapsed,"max_seconds":elapsed})
            values["calls"] += 1
            values["failed_calls"] += int(failed)
            values["total_seconds"] += elapsed
            values["min_seconds"] = min(values["min_seconds"],elapsed)
            values["max_seconds"] = max(values["max_seconds"],elapsed)

    @contextmanager
    def wrap(self, instance, attribute, name):
        missing = object()
        owned = instance.__dict__.get(attribute,missing)
        original = getattr(instance,attribute)
        def measured(*args,**kwargs):
            with self.measure(name):
                return original(*args,**kwargs)
        setattr(instance,attribute,measured)
        try:
            yield
        finally:
            if owned is missing:
                delattr(instance,attribute)
            else:
                setattr(instance,attribute,owned)

    def summary(self):
        return {name:{**values,"mean_seconds":values["total_seconds"]/values["calls"],
                      "mean_after_first_seconds":((values["total_seconds"]-values["first_seconds"])/(values["calls"]-1)
                                                  if values["calls"]>1 else None)}
                for name,values in self.statistics.items()}


@contextmanager
def profile_training(trainer, recorder):
    with ExitStack() as stack:
        for attribute,name in (("step","train_step"),("evaluate","validation"),("save","checkpoint_save")):
            stack.enter_context(recorder.wrap(trainer,attribute,name))
        stack.enter_context(recorder.wrap(trainer.model.segmenter.clusterer,"forward","clustering"))
        yield recorder


def project_training_time(timing, *, steps_per_epoch, epochs=20, checkpoint_every_steps=100,
                          checkpoint_every_seconds=300., safety_factor=2., max_slice_minutes=210):
    if min(steps_per_epoch,epochs,checkpoint_every_steps,max_slice_minutes)<1:
        raise ValueError("Training counts and maximum slice must be positive")
    if not math.isfinite(safety_factor) or safety_factor<1 or not math.isfinite(checkpoint_every_seconds) or checkpoint_every_seconds<=0:
        raise ValueError("Invalid timing projection safety/checkpoint interval")
    for name in ("train_step","validation","checkpoint_save"):
        if timing.get(name,{}).get("calls",0)<1 or timing[name]["failed_calls"]:
            raise ValueError("Projection requires successful training, validation and save measurements")
    steps = steps_per_epoch*epochs
    optimizer_seconds = steps*timing["train_step"]["mean_seconds"]
    validation_calls = epochs+1  # Initial pass plus one per completed epoch.
    validation_seconds = validation_calls*timing["validation"]["mean_seconds"]
    # Upper bound: initial/final/final-epoch, step/time/epoch triggers, every epoch best.
    saves = 3+steps//checkpoint_every_steps+2*epochs+math.ceil((optimizer_seconds+validation_seconds)/checkpoint_every_seconds)
    save_seconds = saves*timing["checkpoint_save"]["mean_seconds"]
    training_seconds = optimizer_seconds+validation_seconds+save_seconds
    # CLI evaluates the final model and restored state once each after the loop.
    post_validation_seconds = 2*timing["validation"]["mean_seconds"]
    guarded_seconds = safety_factor*(training_seconds+post_validation_seconds)
    recommended = min(max_slice_minutes,max(1,math.ceil(guarded_seconds/60)+5))
    return {"epochs":epochs,"steps_per_epoch":steps_per_epoch,"optimizer_steps":steps,
            "optimizer_seconds":optimizer_seconds,"validation_calls":validation_calls,"validation_seconds":validation_seconds,
            "checkpoint_calls_upper_bound":saves,"checkpoint_seconds":save_seconds,
            "estimated_training_loop_seconds":training_seconds,"post_validation_seconds":post_validation_seconds,
            "safety_factor":safety_factor,"guarded_train_and_post_seconds":guarded_seconds,
            "recommended_slice_minutes":recommended,"maximum_slice_minutes":max_slice_minutes,
            "expected_slices":math.ceil(guarded_seconds/(recommended*60)),
            "estimate_scope":"measured compute/validation/save; reserve five minutes for CLI setup and I/O variation"}
