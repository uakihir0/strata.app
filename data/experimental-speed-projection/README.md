# Experimental speed projection (off by default)

This folder contains an experimental 480 KB control vector for Qwen3.8-Flash-Next and its author-provided settings.
The reference feature projects a residual direction after layers 4-44. Its authors describe that direction as related
to refusal behavior, so using it can change which requests the model declines. The feature also changes model outputs
on ordinary text; it is not a speed optimization.

The Mac engine includes the control-vector path, but production inference and the macOS installer are not complete.
This vector is therefore a research asset, not an installable feature in the current checkout. It has not been measured
as a supported Mac product option.

**Origin and license.** The vector was made from Qwen3.8-Flash-Next activations. The Qwen Community License 1.0 of
the model applies to it. The source package's published filenames contain “refusal”; the locally stored bytes are the
same. It was made for the original Qwen3.8-Flash-Next, not Swift 1.5.
