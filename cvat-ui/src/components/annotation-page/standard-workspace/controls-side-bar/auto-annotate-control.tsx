import React, { useState, useCallback } from 'react';
import { connect } from 'react-redux';
import Button from 'antd/lib/button';
import Popover from 'antd/lib/popover';
import message from 'antd/lib/message';
import Typography from 'antd/lib/typography';
import { getCore } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import {
    fetchAnnotationsAsync,
    changeFrameAsync,
} from 'actions/annotation-actions';
import CVATTooltip from 'components/common/cvat-tooltip';
import withVisibilityHandling from './handle-popover-visibility';

const { Text } = Typography;
const core = getCore();

interface Props {
    jobInstance: any;
    frame: number;
    frameIsDeleted: boolean;
    fetchAnnotations: typeof fetchAnnotationsAsync;
    changeFrame: typeof changeFrameAsync;
}

const CustomPopover = withVisibilityHandling(Popover, 'auto-annotate-control');

function mapStateToProps(state: CombinedState): Omit<Props, 'fetchAnnotations' | 'changeFrame'> {
    const {
        annotation: {
            job: { instance: jobInstance },
            player: {
                frame: {
                    number: frame,
                    data: { deleted: frameIsDeleted },
                },
            },
        },
    } = state;

    return {
        jobInstance,
        frame,
        frameIsDeleted,
    };
}

const mapDispatchToProps = {
    fetchAnnotations: fetchAnnotationsAsync,
    changeFrame: changeFrameAsync,
};

function AutoAnnotateControlComponent(props: Props): JSX.Element {
    const {
        jobInstance,
        frame,
        frameIsDeleted,
        fetchAnnotations,
        changeFrame,
    } = props;

    const [loading, setLoading] = useState(false);

    const handleAutoAnnotate = useCallback(async () => {
        if (loading || frameIsDeleted) return;

        setLoading(true);

        try {
            // 1️⃣ Run backend inference
            await core.server.request(
                `/api/jobs/${jobInstance.id}/auto-annotate`,
                {
                    method: 'POST',
                    params: { frame },
                },
            );

            /**
             * 2️⃣ FORCE CANVAS REDRAW
             * This is the ONLY correct refresh mechanism in CVAT 4.2.x
             * - fetches annotations internally
             * - updates Redux player state
             * - redraws canvas immediately
             */
            await jobInstance.annotations.clear({ reload: true });
            changeFrame(frame, false, undefined, true);

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
        changeFrame,
    ]);

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
        <CVATTooltip overlay="Generate inference using YOLOv7" placement="right">
            <CustomPopover placement="right" content={content} trigger="click">
                <Button
                    className="cvat-auto-annotate-control cvat-canvas-control"
                    type="link"
                    disabled={frameIsDeleted}
                >
                    🤖
                </Button>
            </CustomPopover>
        </CVATTooltip>
    );
}

export default connect(
    mapStateToProps,
    mapDispatchToProps,
)(React.memo(AutoAnnotateControlComponent));
