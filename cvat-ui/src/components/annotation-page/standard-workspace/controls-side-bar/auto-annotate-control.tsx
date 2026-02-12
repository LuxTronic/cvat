import React, { useState, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Popover from 'antd/lib/popover';
import message from 'antd/lib/message';
import { RobotOutlined } from '@ant-design/icons';
import Typography from 'antd/lib/typography';
import { getCore } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import GlobalHotKeys from 'utils/mousetrap-react';
import {
    changeFrameAsync,
} from 'actions/annotation-actions';
import { registerComponentShortcuts } from 'actions/shortcuts-actions';
import { ShortcutScope } from 'utils/enums';
import { subKeyMap } from 'utils/component-subkeymap';
import CVATTooltip from 'components/common/cvat-tooltip';
import withVisibilityHandling from './handle-popover-visibility';

const { Text } = Typography;
const core = getCore();

const componentShortcuts = {
    SWITCH_AUTO_ANNOTATE_STANDARD_CONTROLS: {
        name: 'Auto annotate frame',
        description: 'Run automatic annotation for the current frame',
        sequences: ['q'],
        scope: ShortcutScope.STANDARD_WORKSPACE_CONTROLS,
    },
};

registerComponentShortcuts(componentShortcuts);

const CustomPopover = withVisibilityHandling(Popover, 'auto-annotate-control');
function AutoAnnotateControlComponent(): JSX.Element {
    const dispatch = useDispatch<any>();
    const {
        jobInstance,
        frame,
        frameIsDeleted,
        keyMap,
        normalizedKeyMap,
    } = useSelector((state: CombinedState) => ({
        jobInstance: state.annotation.job.instance,
        frame: state.annotation.player.frame.number,
        frameIsDeleted: state.annotation.player.frame.data.deleted,
        keyMap: state.shortcuts.keyMap,
        normalizedKeyMap: state.shortcuts.normalizedKeyMap,
    }));
    const [loading, setLoading] = useState(false);

    const handleAutoAnnotate = useCallback(async () => {
        if (loading || frameIsDeleted) return;

        setLoading(true);

        try {
            if (jobInstance.annotations.hasUnsavedChanges()) {
                await jobInstance.annotations.save();
            }

            await core.server.request(
                `/api/jobs/${jobInstance.id}/auto-annotate`,
                {
                    method: 'POST',
                    params: { frame },
                },
            );

            await jobInstance.annotations.clear({ reload: true });
            dispatch(changeFrameAsync(frame, false, undefined, true));

            message.success('Inference completed successfully');
        } catch (error: any) {
            message.error(
                error?.response?.data?.detail ||
                error?.response?.data?.error ||
                error?.message ||
                'Inference failed',
            );
        } finally {
            setLoading(false);
        }
    }, [
        jobInstance.id,
        frame,
        frameIsDeleted,
        loading,
        dispatch,
        jobInstance,
    ]);

    const handlers: Record<keyof typeof componentShortcuts, (event?: KeyboardEvent) => void> = {
        SWITCH_AUTO_ANNOTATE_STANDARD_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            void handleAutoAnnotate();
        },
    };

    const content = (
        <div style={{ textAlign: 'center', minWidth: 200 }}>
            <Text>Generate inference for current frame</Text>
            <br /><br />
            <Button
                type="primary"
                onClick={handleAutoAnnotate}
                disabled={loading || frameIsDeleted}
                loading={loading}
            >
                Generate inference
            </Button>
        </div>
    );

    return (
        <>
            <GlobalHotKeys keyMap={subKeyMap(componentShortcuts, keyMap)} handlers={handlers} />
            <CVATTooltip
                title={`Generate inference using YOLOv7 ${normalizedKeyMap.SWITCH_AUTO_ANNOTATE_STANDARD_CONTROLS}`}
                placement='right'
            >
                <CustomPopover placement='right' content={content} trigger='click'>
                    <Button
                        className="cvat-auto-annotate-control cvat-canvas-control"
                        type="link"
                        disabled={frameIsDeleted}>
                        <RobotOutlined />
                    </Button>
                </CustomPopover>
            </CVATTooltip>
        </>
    );
}

export default React.memo(AutoAnnotateControlComponent);
