# ec-tracking-analysis

Please visit the wiki for more information https://github.com/gerhardt-lab/ec-tracking-analysis/wiki .

## Expected Input Format

This pipeline processes **TrackMate CSV exports**. Input files must contain the following columns:

| Column | Full Name | Units |
|--------|-----------|-------|
| LABEL | Label | |
| ID | Spot ID | |
| TRACK_ID | Track ID | |
| QUALITY | Quality | |
| POSITION_X | X | μm |
| POSITION_Y | Y | μm |
| POSITION_Z | Z | μm |
| POSITION_T | T | sec |
| FRAME | Frame | |
| RADIUS | Radius | μm |
| VISIBILITY | Visibility | |
| MANUAL_SPOT_COLOR | Manual spot color | |
| MEAN_INTENSITY_CH1 | Mean intensity ch1 | counts |
| MEDIAN_INTENSITY_CH1 | Median intensity ch1 | counts |
| MIN_INTENSITY_CH1 | Min intensity ch1 | counts |
| MAX_INTENSITY_CH1 | Max intensity ch1 | counts |
| TOTAL_INTENSITY_CH1 | Sum intensity ch1 | counts |
| STD_INTENSITY_CH1 | Std intensity ch1 | counts |
| CONTRAST_CH1 | Contrast ch1 | |
| SNR_CH1 | Signal/Noise ratio ch1 | |
| ELLIPSE_X0 | Ellipse center x0 | μm |
| ELLIPSE_Y0 | Ellipse center y0 | μm |
| ELLIPSE_MAJOR | Ellipse long axis | μm |
| ELLIPSE_MINOR | Ellipse short axis | μm |
| ELLIPSE_THETA | Ellipse angle | radians |
| ELLIPSE_ASPECTRATIO | Ellipse aspect ratio | |
| AREA | Area | μm² |
| PERIMETER | Perimeter | μm |
| CIRCULARITY | Circularity | |
| SOLIDITY | Solidity | |
| SHAPE_INDEX | Shape index | |

**Note:** Coordinate values (POSITION_X, POSITION_Y) are in micrometers as calibrated by TrackMate.

