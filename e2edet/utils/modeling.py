from e2edet.utils.general import filter_grads


def get_parameters(module, lr_multi=None, lr_module=[], lr_except=["backbone"]):
    param_optimizer = list(module.named_parameters())

    optimizer_grouped_parameters = [
        {
            "params": filter_grads(
                [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in (lr_except + lr_module))
                ]
            ),
        },
        {
            "params": filter_grads(
                [
                    p
                    for n, p in param_optimizer
                    if (
                        any(nd in n for nd in lr_module)
                        and not any(nd in n for nd in lr_except)
                    )
                ]
            ),
            "lr_multi": lr_multi if lr_multi is not None else 1.0,
        },
    ]

    return optimizer_grouped_parameters
