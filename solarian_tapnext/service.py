from solarian.services import Service


class TapNextService(Service):
    """TAPNext++: points seeded on a frame, followed through the next ones."""

    server_module = "solarian_tapnext.server"
    default_port = 8767
    checkpoint_arg = "--checkpoint"

    @classmethod
    def modal_spec(cls):
        from solarian.core.ai.system1.policy_trainer import ModalSpec

        # The model code is vendored in the package; the weights are the
        # checkpoint modal-serve is given.
        return ModalSpec(
            apt_packages=["libgl1", "libglib2.0-0"],
            pip_packages=["torch", "torchvision", "einops", "fastapi", "uvicorn"],
            local_python_packages=["solarian_tapnext"],
        )
