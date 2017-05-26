from collections import namedtuple

Model = namedtuple("Model",
	"inputs outputs loss train_op global_step "
	"figsize training grad_norm max_steps")
