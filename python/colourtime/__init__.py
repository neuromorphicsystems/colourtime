import re
import typing

import numpy
import numpy.lib.recfunctions
import numpy.typing
import PIL.Image

from . import colourtime


def convert(
    begin: typing.Optional[int],
    end: typing.Optional[int],
    width: int,
    height: int,
    decoder: typing.Generator[numpy.ndarray, None, None],
    colormap: typing.Callable[
        [numpy.typing.NDArray[numpy.float64]], numpy.typing.NDArray[numpy.float64]
    ],
    time_mapping: typing.Callable[
        [numpy.typing.NDArray[numpy.uint64]], numpy.typing.NDArray[numpy.float64]
    ],
    alpha: float,
    background_colour: tuple[float, float, float, float],
) -> PIL.Image.Image:
    image = numpy.zeros((width, height, 4), dtype=numpy.float64)
    image[:, :] = background_colour
    for packet in decoder:
        if begin is not None and packet["t"][-1] < begin:
            continue
        if end is not None and packet["t"][0] >= end:
            break
        if (
            begin is not None
            and end is not None
            and packet["t"][0] < begin
            and packet["t"][-1] >= end
        ):
            events = packet[numpy.logical_and(packet["t"] >= begin, packet["t"] < end)]
        elif begin is not None and packet["t"][0] < begin:
            events = packet[packet["t"] >= begin]
        elif end is not None and packet["t"][-1] >= end:
            events = packet[packet["t"] < end]
        else:
            events = packet
        xy = numpy.lib.recfunctions.repack_fields(events[["x", "y"]]).view(  # type: ignore
            (numpy.uint16, 2)
        )
        colours = colormap(time_mapping(events["t"]))
        colourtime.stack(image, xy, colours, alpha)  # type: ignore
    return PIL.Image.fromarray(
        numpy.round(image * 255.0).astype(numpy.uint8),
        mode="RGBA",
    ).transpose(
        PIL.Image.Transpose.ROTATE_90  # type: ignore
    )


def generate_cyclic_time_mapping(duration: int, begin: int):
    def cyclic_time_mapping(
        timestamps: numpy.typing.NDArray[numpy.uint64],
    ) -> numpy.typing.NDArray[numpy.float64]:
        return ((timestamps - begin) % duration) / float(duration)

    return cyclic_time_mapping


def generate_linear_time_mapping(begin: int, end: int):
    assert begin < end

    def linear_time_mapping(
        timestamps: numpy.typing.NDArray[numpy.uint64],
    ) -> numpy.typing.NDArray[numpy.float64]:
        return (timestamps - begin) / (end - begin)

    return linear_time_mapping


def find_begin_and_end(
    decoder: typing.Generator[numpy.ndarray, None, None],
    find_end: bool,
) -> tuple[int, typing.Optional[int]]:
    begin = None
    end = None
    if find_end:
        for packet in decoder:
            if begin is None:
                begin = packet["t"][0]
            end = packet["t"][-1]
        if begin is None:
            begin = 0
        if end is None:
            end = begin
        end += 1
    else:
        packet = next(decoder, None)
        if packet is None:
            begin = 0
        else:
            begin = packet["t"][0]
    assert begin is not None
    return (begin, end)


TIMECODE_PATTERN = re.compile(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$")


def timecode(value: str) -> int:
    if value.isdigit():
        return int(value)
    match = TIMECODE_PATTERN.match(value)
    if match is None:
        import argparse

        raise argparse.ArgumentTypeError(
            "expected an integer or a timecode (12:34:56.789000)"
        )
    result = (
        int(match[1]) * 3600000000 + int(match[2]) * 60000000 + int(match[3]) * 1000000
    )
    if match[4] is not None:
        fraction_string: str = match[4]
        if len(fraction_string) == 6:
            result += int(fraction_string)
        elif len(fraction_string) < 6:
            result += int(fraction_string + "0" * (6 - len(fraction_string)))
        else:
            result += round(float("0." + fraction_string) * 1e6)
    return result


def main():
    import argparse
    import pathlib

    import event_stream
    import matplotlib
    import matplotlib.colors

    parser = argparse.ArgumentParser(
        description="Convert an Event Stream file into a colourtime plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="path to the input Event Stream (.es) file")
    parser.add_argument(
        "--begin",
        "-b",
        type=timecode,
        help="ignore events with timestamps strictly smaller than begin (timecode)",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=timecode,
        help="ignore events with timestamps larger than or equal to end (timecode)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to the output PNG file (generated from the input path if not provided)",
    )
    parser.add_argument(
        "--colormap",
        "-c",
        default="viridis",
        help="colormap (see https://matplotlib.org/stable/tutorials/colors/colormaps.html)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.1,
        help="event opaacity in ]0, 1]",
    )
    parser.add_argument(
        "--cycle",
        "-y",
        type=timecode,
        help="enable cyclic time mapping with the given duration in seconds",
    )
    parser.add_argument(
        "--background-colour",
        "-k",
        default="#191919ff",
        help="background colour (RGB or RGBA)",
    )
    parser.add_argument(
        "--png-compression-level",
        "-p",
        type=int,
        default=6,
        help="PNG compression level in [0, 9] (0 - fastest, 9 - smallest)",
    )

    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()

    if args.output is None:
        name = f"{input_path.stem}_begin={args.begin}_end={args.end}_colormap={args.colormap}_alpha={args.alpha}"
        if args.cycle is not None:
            name += f"_cycle={args.cycle}"
        print(name)
        output = input_path.parent / f"{name}.png"
    else:
        output = pathlib.Path(args.output).resolve()

    if args.cycle:
        if args.begin is None:
            with event_stream.Decoder(args.input) as decoder:
                begin = find_begin_and_end(decoder=decoder, find_end=False)[0]
        else:
            begin = args.begin
        time_mapping = generate_cyclic_time_mapping(duration=args.cycle, begin=begin)
    else:
        if args.begin is None or args.end is None:
            with event_stream.Decoder(args.input) as decoder:
                stream_begin, stream_end = find_begin_and_end(
                    decoder=decoder, find_end=args.end is None
                )
            if args.begin is None:
                begin = stream_begin
            else:
                begin = args.begin
            if args.end is None:
                assert stream_end is not None
                end = stream_end
            else:
                end = args.end
        else:
            begin = args.begin
            end = args.end
        time_mapping = generate_linear_time_mapping(begin=begin, end=end)

    with event_stream.Decoder(args.input) as decoder:
        image = convert(
            begin=args.begin,
            end=args.end,
            width=decoder.width,
            height=decoder.height,
            decoder=decoder,
            colormap=matplotlib.colormaps[args.colormap],  # type: ignore
            time_mapping=time_mapping,
            alpha=args.alpha,
            background_colour=matplotlib.colors.to_rgba(args.background_colour),
        )

    image.save(str(output), compress_level=args.png_compression_level)


if __name__ == "__main__":
    main()
