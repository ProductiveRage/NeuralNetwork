using System.Collections.Immutable;

namespace NeuralNetwork;

public static class ImmutableArrayExtensions
{
    public static ImmutableArray<TResult> Select<TSource, TResult>(
        this ImmutableArray<TSource> source,
        Func<TSource, TResult> selector) =>
            Enumerable.Select(source, selector).ToImmutableArray();

    public static ImmutableArray<TResult> Select<TSource, TResult>(
        this ImmutableArray<TSource> source,
        Func<TSource, int, TResult> selector) =>
            Enumerable.Select(source, selector).ToImmutableArray();

    public static ImmutableArray<TResult> Zip<TFirst, TSecond, TResult>(
        this ImmutableArray<TFirst> source,
        IEnumerable<TSecond> with,
        Func<TFirst, TSecond, TResult> selector) =>
            Enumerable.Zip(source, with, selector).ToImmutableArray();
}