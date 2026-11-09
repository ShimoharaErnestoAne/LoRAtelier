from comfy_api.latest import io

PPMRefCond = io.Custom("PPM_REF_COND")


class ReferenceCondPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReferenceCondPPM",
            display_name="Reference Cond (PPM)",
            category="conditioning",
            search_aliases=["reference list", "combine references", "character reference"],
            description="Wraps reference image, strength and caption into PPM_REF_COND.",
            inputs=[
                PPMRefCond.Input(
                    "ppm_ref_cond",
                    optional=True,
                    tooltip="Optional previous Reference Cond for chaining multiple references.",
                ),
                io.Image.Input("image"),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-3.0,
                    max=3.0,
                    step=0.05,
                ),
                io.String.Input(
                    "caption",
                    multiline=True,
                    default="",
                    placeholder="Optional caption for reference.",
                    tooltip="Optional caption for reference.",
                ),
            ],
            outputs=[PPMRefCond.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        image, strength, caption = kwargs["image"], float(kwargs["strength"]), kwargs["caption"]
        refs = list(kwargs["ppm_ref_cond"]) if kwargs.get("ppm_ref_cond") else []
        if strength > 0.0:
            refs += [
                {"image": image[index : index + 1], "strength": strength, "caption": caption}
                for index in range(image.shape[0])
            ]
        return io.NodeOutput(refs)


class ReferenceCondUnwrapPPM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ReferenceCondUnwrapPPM",
            display_name="Reference Cond Unwrap (PPM)",
            category="conditioning",
            search_aliases=["reference list", "split references", "character reference"],
            description="Unwraps PPM_REF_COND into reference image, strength and caption.",
            inputs=[
                PPMRefCond.Input("ppm_ref_cond"),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=63,
                    tooltip="Index of image inside PPM_REF_COND to unwrap.",
                ),
            ],
            outputs=[
                io.Image.Output(),
                io.Float.Output(),
                io.String.Output(),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        refs: list[dict] = kwargs["ppm_ref_cond"]
        if not refs:
            raise ValueError("ppm_ref_cond holds no references to output")
        entry = refs[min(int(kwargs["index"]), len(refs) - 1)]
        return io.NodeOutput(entry["image"], float(entry.get("strength", 1.0)), str(entry.get("caption", "")))


NODES = [ReferenceCondPPM, ReferenceCondUnwrapPPM]
