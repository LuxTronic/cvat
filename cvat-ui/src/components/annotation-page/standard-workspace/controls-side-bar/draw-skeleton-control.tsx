import React from 'react';
import { useSelector } from 'react-redux';
import Popover from 'antd/lib/popover';
import Icon from '@ant-design/icons';

import { Canvas } from 'cvat-canvas-wrapper';
import { Canvas3d } from 'cvat-canvas3d-wrapper';
import { ShapeType } from 'cvat-core-wrapper';

import { SkeletonIcon } from 'icons';

import { CombinedState } from 'reducers';
import CVATTooltip from 'components/common/cvat-tooltip';
import DrawShapePopoverContainer from 'containers/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover';
import withVisibilityHandling from './handle-popover-visibility';

export interface Props {
    canvasInstance: Canvas | Canvas3d;
    isDrawing: boolean;
    disabled: boolean;
}

const CustomPopover = withVisibilityHandling(Popover, 'draw-skeleton');
function DrawSkeletonControl(props: Props): JSX.Element {
    const { canvasInstance, isDrawing, disabled } = props;
    const { normalizedKeyMap } = useSelector((state: CombinedState) => state.shortcuts);
    const dynamicPopoverProps = isDrawing ? {
        overlayStyle: {
            display: 'none',
        },
    } : {};

    const dynamicIconProps = isDrawing ? {
        className: 'cvat-draw-skeleton-control cvat-active-canvas-control',
        onClick: (): void => {
            canvasInstance.draw({ enabled: false });
        },
    } : {
        className: 'cvat-draw-skeleton-control',
    };

    return disabled ? (
        <Icon className='cvat-draw-skeleton-control cvat-disabled-canvas-control' component={SkeletonIcon} />
    ) : (
        <CustomPopover
            {...dynamicPopoverProps}
            overlayClassName='cvat-draw-shape-popover'
            placement='right'
            content={<DrawShapePopoverContainer shapeType={ShapeType.SKELETON} />}
        >
            <CVATTooltip
                title={`Draw a skeleton. ${canvasInstance instanceof Canvas ?
                    normalizedKeyMap.SWITCH_DRAW_MODE_STANDARD_CONTROLS :
                    normalizedKeyMap.SWITCH_DRAW_MODE_STANDARD_3D_CONTROLS} repeats last drawing action`}
                placement='right'
            >
                <Icon {...dynamicIconProps} component={SkeletonIcon} />
            </CVATTooltip>
        </CustomPopover>
    );
}

Object.assign(DrawSkeletonControl, { displayName: 'DrawSkeletonControl' });
export default React.memo(DrawSkeletonControl);
