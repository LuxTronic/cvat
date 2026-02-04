import React, { useState, useCallback } from 'react';
import { connect } from 'react-redux';
import Button from 'antd/lib/button';
import Popover from 'antd/lib/popover';
import message from 'antd/lib/message';
import Typography from 'antd/lib/typography';
import { getCore } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { fetchAnnotationsAsync } from 'actions/annotation-actions';
import CVATTooltip from 'components/common/cvat-tooltip';
import withVisibilityHandling from './handle-popover-visibility';

const { Text } = Typography;
const core = getCore();

interface Props {
    jobInstance: any;
    frame: number;
    frameIsDeleted: boolean;
}

interface DispatchToProps {
    fetchAnnotations: typeof fetchAnnotationsAsync;
}

const CustomPopover = withVisibilityHandling(Popover, 'auto-annotate-control');

function mapStateToProps(state: CombinedState): Props {
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

    return { jobInstance, frame, frameIsDeleted };
}

const mapDispatchToProps: DispatchToProps = {
    fetchAnnotations: fetchAnnotationsAsync,
};

function AutoAnnotateControlComponent(props: Props & DispatchToProps): JSX.Element {
    const { jobInstance, frame, frameIsDeleted, fetchAnnotations } = props;
    const [loading, setLoading] = useState(false);

    const handleAutoAnnotate = useCallback(async () => {
        if (loading || frameIsDeleted) return;

        setLoading(true);

        try {
            await core.server.request(
                `/api/jobs/${jobInstance.id}/auto-annotate`,
                {
                    method: 'POST',
                    params: { frame },
                },
            );

            message.success('Inference completed successfully');

            // 🔑 This updates Redux → canvas redraws
            fetchAnnotations();
        } catch (error: any) {
            message.error(error?.message || 'Inference failed');
        } finally {
            setLoading(false);
        }
    }, [jobInstance, frame, frameIsDeleted, loading, fetchAnnotations]);

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
