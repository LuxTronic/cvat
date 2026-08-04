// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import Popover from 'antd/lib/popover';
import Icon from '@ant-design/icons';

import { CombinedState } from 'reducers';
import CVATTooltip from 'components/common/cvat-tooltip';
import { useSelector } from 'react-redux';

import { Canvas } from 'cvat-canvas-wrapper';
import { BrushIcon } from 'icons';
import { ShapeType } from 'cvat-core-wrapper';

import DrawShapePopoverContainer from 'containers/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover';
import withVisibilityHandling from './handle-popover-visibility';

export interface Props {
    canvasInstance: Canvas;
    isDrawing: boolean;
    disabled?: boolean;
}

const CustomPopover = withVisibilityHandling(Popover, 'draw-mask');
function DrawPointsControl(props: Props): JSX.Element {
    const { canvasInstance, isDrawing, disabled } = props;

    const { normalizedKeyMap } = useSelector((state: CombinedState) => state.shortcuts);

    const dynamicPopoverProps = isDrawing ? {
        overlayStyle: {
            display: 'none',
        },
    } : {};

    const dynamicIconProps = isDrawing ? {
        className: 'cvat-draw-mask-control cvat-active-canvas-control',
        onClick: (): void => {
            canvasInstance.draw({ enabled: false });
        },
    } : {
        className: 'cvat-draw-mask-control',
    };

    return disabled ? (
        <Icon className='cvat-draw-mask-control cvat-disabled-canvas-control' component={BrushIcon} />
    ) : (
        <CVATTooltip
            title={`Draw a mask. ${normalizedKeyMap.SWITCH_DRAW_MODE_STANDARD_CONTROLS} repeats last drawing action`}
            placement='right'
        >
            <CustomPopover
                {...dynamicPopoverProps}
                overlayClassName='cvat-draw-shape-popover'
                placement='right'
                content={<DrawShapePopoverContainer shapeType={ShapeType.MASK} />}
            >
                <Icon {...dynamicIconProps} component={BrushIcon} />
            </CustomPopover>
        </CVATTooltip>
    );
}

export default React.memo(DrawPointsControl);
