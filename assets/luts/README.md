# Your own LUTs

Drop any **3D `.cube` LUT** here (exported from DaVinci Resolve, Premiere, Final Cut, or a LUT pack you own).
Restart the app and it appears as a look named `lut:<file name>`. It works in Look Lab and the AI editors can choose it.

- Reelsmith applies the LUT after its own per-shot **color matching**, so mismatched shots are balanced first.
- LUTs designed for a *log* camera profile (S-Log, V-Log, Apple Log…) expect log footage. For normal phone video,
  use LUTs made for Rec.709.
- These files are committed to git like the rest of the project. Remove them before pushing if their licence doesn't allow sharing.
