# Dark workbench design QA

final result: passed

Source visual truth: `design/selected-option-3.png` (third displayed ideation result), lower frame `design/source-screen.png`. Implementation: `design/desktop.png` and `design/mobile.png`.

Desktop CSS viewport 1440 × 1024, devicePixelRatio 1. Source frame 1053 × 747; browser PNG 1425 × 1013. Mobile CSS viewport 390 × 844; full-page PNG 375 × 1835. Captures exclude the scrollbar. Both reference and desktop were fit with preserved aspect ratios into equal 720 × 512 comparison slots; no stretched imagery or device chrome. State: 4× selected, illustrative input on both sides, no trained weights, download disabled.

Full-view evidence: `design/comparison.png`. Focused tool-panel evidence: `design/comparison-detail.png`. Both were opened together as source/implementation comparison inputs before this report.

## Findings and comparison history

- Initial P2: Gradio HTML/main padding disrupted the compact canvas-first layout. Removed duplicate padding and preserved the reference's tool-panel/canvas proportions.
- Initial P2: ImageSlider's inline 500px image limit letterboxed the photograph within the 640px workspace. Removed the inline maximum with scoped CSS and sized both images identically while preserving image aspect ratio. Final desktop screenshot fills the intended canvas.
- Initial P2: the pre-training notice appeared behind the image at mobile width. Added an explicit notice stacking level and visible overflow. Final mobile capture and DOM verify the readable notice above the photograph (z-index 15).
- Initial usability finding: raw generated sample would exceed the 16MP output cap at 4×. Bound the illustrative sample to 960 × 640 before loading it, yielding 956 × 640 input and 3824 × 2560 target. Real uploads retain their original dimensions and output safeguards.
- Final comparison: no remaining P0/P1/P2 findings.

## Required fidelity surfaces

- Typography: locally bundled Inter with compact tool labels, restrained heading scale and legible status/dimension text. No remote font loading.
- Layout: narrow input/tool panel plus large slider canvas; dimensions/time below. Stacked mobile layout with no horizontal overflow (document/client widths both 375px).
- Colors: graphite #111416 and #171b1e surfaces, mint #83eacb action/selected accents, subtle blue-gray dividers and clear disabled download state.
- Images/icons: generated natural waterfall/forest photograph matches the reference's subject, density and palette. Unmodified Tabler icon assets and bundled font licenses included.
- Copy/content: both slider sides are explicitly original input previews until a genuine reconstruction exists. The slider label, overlay, status and disabled download avoid representing illustrations as model outputs. Target dimensions are distinguished from generated output dimensions.

## Interactions and validation

Browser tested divider dragging (clip position changed from 50% to 68.63%), 2×/4× selection and target dimension updates, sample reload, disabled download, missing-checkpoint error and in-page How it works navigation. Input/scale changes reset output and download state. The slider's built-in preview-download link is disabled. Three targeted interface checks passed; final two presentation tests passed again after lazy imports/sample limits. Lint, HTTP startup, CPU Space ZIP integrity/current assets checks passed. UI initialization independently verified PyTorch remains unimported. Expected Gradio error responses occurred during intentional missing-weight checks; no unexpected browser asset/runtime error was observed.

## Intentional differences and limits

The photograph is regenerated, not an exact copy. Both sides are identical input rather than the mock's artificial blur/sharp illustration, honoring the requirement for genuine trained outputs. Native Gradio slider handle and segmented-control styling differ from the drawn controls (accepted P3). The divider was pointer-tested; a comprehensive keyboard/screen-reader/WCAG audit remains outside this check. Actual reconstruction, inference timing and neural PNG downloads await approved training/export.

Implementation checklist: complete. Training and publication: not started.

