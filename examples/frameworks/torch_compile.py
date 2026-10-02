"""CPU checks for capture reuse, shape guards, graph breaks and AOT gradients."""

import argparse

import torch


def polynomial(x):
    return x * x + 3 * x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inductor",
        action="store_true",
        help="also generate CPU code; requires a compatible C++ compiler",
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    dtype = torch.float64

    x = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype, requires_grad=True)
    compiled = torch.compile(polynomial, backend="aot_eager", fullgraph=True)
    y = compiled(x)
    torch.testing.assert_close(y, torch.tensor([-2.0, 0.0, 10.0], dtype=dtype))
    y.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() + 3)

    captures = []

    def counting_backend(graph_module, example_inputs):
        captures.append(graph_module)
        return graph_module.forward

    # This backend observes Dynamo only; it does not optimize device kernels.
    torch.compiler.reset()
    static = torch.compile(polynomial, backend=counting_backend, fullgraph=True, dynamic=False)
    for size in (2, 2, 3):
        value = torch.arange(size, dtype=dtype)
        torch.testing.assert_close(static(value), polynomial(value))
    assert len(captures) == 2, len(captures)
    static_captures = len(captures)

    torch.compiler.reset()
    captures.clear()
    dynamic = torch.compile(polynomial, backend=counting_backend, fullgraph=True, dynamic=True)
    for size in (2, 3, 5):
        value = torch.arange(size, dtype=dtype)
        torch.testing.assert_close(dynamic(value), polynomial(value))
    assert len(captures) == 1, len(captures)
    dynamic_captures = len(captures)
    value = torch.arange(5, dtype=torch.float32)
    torch.testing.assert_close(dynamic(value), polynomial(value))
    assert len(captures) == 2, "dtype change must select a new compiled variant"

    @torch.compiler.disable
    def python_region(value):
        return value + 1

    def segmented(value):
        return python_region(value * 2) * 3

    torch.compiler.reset()
    captures.clear()
    partial = torch.compile(segmented, backend=counting_backend)
    value = torch.tensor([1.0, 2.0], dtype=dtype)
    torch.testing.assert_close(partial(value), torch.tensor([9.0, 15.0], dtype=dtype))
    assert len(captures) == 2, len(captures)
    torch.compiler.reset()
    strict = torch.compile(segmented, backend=counting_backend, fullgraph=True)
    try:
        strict(value)
    except torch._dynamo.exc.Unsupported:
        pass
    else:
        raise AssertionError("fullgraph accepted an explicitly disabled region")

    if args.inductor:
        torch.compiler.reset()
        optimized = torch.compile(polynomial, backend="inductor", fullgraph=True)
        value = torch.randn(17, dtype=dtype, requires_grad=True)
        reference = value.detach().clone().requires_grad_()
        actual = optimized(value)
        expected = polynomial(reference)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(value.grad, reference.grad, rtol=1e-10, atol=1e-10)
        print("inductor: generated CPU forward/backward agree with eager")

    torch.compiler.reset()
    print(
        f"torch.compile: AOT gradients, static={static_captures}, "
        f"dynamic={dynamic_captures}, dtype guard and graph breaks OK "
        f"(torch {torch.__version__})"
    )


if __name__ == "__main__":
    main()
