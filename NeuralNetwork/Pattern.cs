using System.Collections.Immutable;

namespace NeuralNetwork;

public sealed class Pattern
{
	public Pattern(ImmutableArray<double> inputs, ImmutableArray<double> outputs)
	{
		if (!inputs.Any())
			throw new ArgumentException("There must be at least one input in a Pattern");
		if (!outputs.Any())
			throw new ArgumentException("There must be at least one output in a Pattern");

		Inputs = inputs;
		Outputs = outputs;
		if (Outputs.Any(o => (o < 0) || (o > 1)))
			throw new ArgumentException("Outputs must all be in the range 0-1 (inclusive)");
	}

	/// <summary>
	/// All values will be in the range 0-1 (inclusive)
	/// </summary>
	public ImmutableArray<double> Inputs { get; }

	/// <summary>
	/// All values will be in the range 0-1 (inclusive)
	/// </summary>
	public ImmutableArray<double> Outputs { get; }
}