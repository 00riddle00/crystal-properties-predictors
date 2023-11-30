class AverageMeter:
    """Compute and stores the average and current value."""

    def __init__(self, name: str):
        self.name = name
        self.val: float | None = None
        self.count: int | None = None
        self.sum: float | None = None
        self.avg: float | None = None
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count
