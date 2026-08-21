// Copyright (C) 2020-2022 Intel Corporation
//
// SPDX-License-Identifier: MIT

import React from 'react';
import Popover from 'antd/lib/popover';
import Icon from '@ant-design/icons';

import { Canvas } from 'cvat-canvas-wrapper';
import { PolygonIcon } from 'icons';
import { ShapeType } from 'cvat-core-wrapper';

import CVATTooltip from 'components/common/cvat-tooltip';
import DrawShapePopoverContainer from 'containers/annotation-page/standard-workspace/controls-side-bar/draw-shape-popover';
import withVisibilityHandling from './handle-popover-visibility';
import useDrawShapeTooltip from './use-draw-shape-tooltip';

export interface Props {
    canvasInstance: Canvas;
    isDrawing: boolean;
    disabled?: boolean;
}

const CustomPopover = withVisibilityHandling(Popover, 'draw-polygon');
function DrawPolygonControl(props: Props): JSX.Element {
    const tooltip = useDrawShapeTooltip('Draw a polygon');
    const { canvasInstance, isDrawing, disabled } = props;
    const dynamicPopoverProps = isDrawing ? {
        overlayStyle: {
            display: 'none',
        },
    } : {};

    const dynamicIconProps = isDrawing ? {
        className: 'cvat-draw-polygon-control cvat-active-canvas-control',
        onClick: (): void => {
            canvasInstance.draw({ enabled: false });
        },
    } : {
        className: 'cvat-draw-polygon-control',
    };

    return disabled ? (
        <Icon className='cvat-draw-polygon-control cvat-disabled-canvas-control' component={PolygonIcon} />
    ) : (
        <CustomPopover
            {...dynamicPopoverProps}
            overlayClassName='cvat-draw-shape-popover'
            placement='right'
            content={<DrawShapePopoverContainer shapeType={ShapeType.POLYGON} />}
        >
            <CVATTooltip title={tooltip} placement='right'>
                <Icon {...dynamicIconProps} component={PolygonIcon} />
            </CVATTooltip>
        </CustomPopover>
    );
}

Object.assign(DrawPolygonControl, { displayName: 'DrawPolygonControl' });
export default React.memo(DrawPolygonControl);
