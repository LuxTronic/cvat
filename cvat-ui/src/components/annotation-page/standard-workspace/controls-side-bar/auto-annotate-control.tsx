import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Popover from 'antd/lib/popover';
import Input from 'antd/lib/input';
import Select from 'antd/lib/select';
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
import { withUIBasePath } from 'utils/base-path';
import withVisibilityHandling from './handle-popover-visibility';

const { Text } = Typography;
const { TextArea } = Input;
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
        labels,
        frame,
        frameIsDeleted,
        keyMap,
        normalizedKeyMap,
    } = useSelector((state: CombinedState) => ({
        jobInstance: state.annotation.job.instance,
        labels: state.annotation.job.labels,
        frame: state.annotation.player.frame.number,
        frameIsDeleted: state.annotation.player.frame.data.deleted,
        keyMap: state.shortcuts.keyMap,
        normalizedKeyMap: state.shortcuts.normalizedKeyMap,
    }));
    const [loading, setLoading] = useState(false);
    const [modelOptionsLoading, setModelOptionsLoading] = useState(false);
    const [modelOptions, setModelOptions] = useState<any[]>([]);
    const [selectedModelID, setSelectedModelID] = useState('');
    const [modelPrompt, setModelPrompt] = useState('');
    const [contextFrameText, setContextFrameText] = useState('');
    const labelNames = useMemo(
        () => (labels || [])
            .map((label: any): string => String(label?.name || '').trim())
            .filter((name: string): boolean => Boolean(name.length)),
        [labels],
    );
    const selectedModel = useMemo(
        () => modelOptions.find((model: any): boolean => model?.id === selectedModelID),
        [modelOptions, selectedModelID],
    );

    useEffect(() => {
        let cancelled = false;

        async function loadModelOptions(): Promise<void> {
            if (!jobInstance?.id) {
                setModelOptions([]);
                setSelectedModelID('');
                return;
            }

            setModelOptionsLoading(true);
            try {
                const response = await core.server.request(
                    withUIBasePath(`/api/jobs/${jobInstance.id}/auto-annotate-models`),
                    {
                        method: 'GET',
                    },
                );

                if (!cancelled) {
                    const models = response?.data?.models || [];
                    setModelOptions(models);
                    setSelectedModelID((current): string => (
                        current && models.some((model: any): boolean => model?.id === current) ?
                            current :
                            ''
                    ));
                }
            } catch {
                if (!cancelled) {
                    setModelOptions([]);
                    setSelectedModelID('');
                }
            } finally {
                if (!cancelled) {
                    setModelOptionsLoading(false);
                }
            }
        }

        void loadModelOptions();

        return () => {
            cancelled = true;
        };
    }, [jobInstance?.id]);

    const parseContextFrame = useCallback((): number[] | null => {
        const trimmed = contextFrameText.trim();
        if (!trimmed.length) {
            return [];
        }

        const parsed = Number(trimmed);
        if (!Number.isInteger(parsed) || parsed < 0) {
            return null;
        }

        if (parsed === frame) {
            return null;
        }

        return [parsed];
    }, [contextFrameText, frame]);

    const reloadAnnotations = useCallback(async () => {
        await jobInstance.annotations.clear({ reload: true });
        dispatch(changeFrameAsync(frame, false, undefined, true));
    }, [dispatch, frame, jobInstance]);

    const handleAutoAnnotate = useCallback(async () => {
        if (loading || frameIsDeleted) return;

        setLoading(true);

        try {
            if (jobInstance.annotations.hasUnsavedChanges()) {
                await jobInstance.annotations.save();
            }

            const params: Record<string, string | number> = { frame };
            if (selectedModelID) {
                params.model_id = selectedModelID;
            }

            await core.server.request(
                withUIBasePath(`/api/jobs/${jobInstance.id}/auto-annotate`),
                {
                    method: 'POST',
                    params,
                },
            );

            await reloadAnnotations();
            message.success(
                selectedModel?.name ?
                    `Inference completed with ${selectedModel.name}` :
                    'Inference completed successfully',
            );
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
        frame,
        frameIsDeleted,
        jobInstance,
        loading,
        reloadAnnotations,
        selectedModel,
        selectedModelID,
    ]);

    const handleModelAnnotate = useCallback(async () => {
        if (loading || frameIsDeleted || !modelPrompt.trim()) return;

        const contextFrameIds = parseContextFrame();
        if (contextFrameIds === null) {
            message.error('Context frame must be one integer and different from current frame');
            return;
        }

        setLoading(true);

        try {
            if (jobInstance.annotations.hasUnsavedChanges()) {
                await jobInstance.annotations.save();
            }

            const response = await core.server.request(
                withUIBasePath('/api/model/annotate'),
                {
                    method: 'POST',
                    data: {
                        task_id: jobInstance.taskId,
                        frame_ids: [frame],
                        context_frame_ids: contextFrameIds,
                        prompt: modelPrompt.trim(),
                    },
                },
            );


            const rqID = response?.data?.rq_id;
            if (!rqID) {
                throw new Error('Model request did not return rq_id');
            }

            const finalRequest = await core.requests.listen(rqID, {
                callback: () => {},
            });

            await reloadAnnotations();

            const requestMessage = (finalRequest.message || '').trim();
            if (requestMessage.startsWith('Model response:')) {
                const aiText = requestMessage.replace(/^Model response:\s*/i, '').trim();
                if (aiText.length) {
                    message.info(aiText, 8);
                } else {
                    message.success('Model request completed');
                }
            } else {
                message.success('Model annotations generated');
            }
        } catch (error: any) {
            message.error(
                error?.response?.data?.detail ||
                error?.response?.data?.error ||
                error?.message ||
                'Model annotation failed',
            );
        } finally {
            setLoading(false);
        }
    }, [
        frame,
        frameIsDeleted,
        modelPrompt,
        jobInstance,
        loading,
        parseContextFrame,
        reloadAnnotations,
    ]);

    const handleAutofillPromptWithLabels = useCallback(() => {
        if (!labelNames.length) {
            message.warning('No task labels found to build a prompt');
            return;
        }

        setModelPrompt(`Annotate the objects in the image where the labels are: ${labelNames.join(', ')}.`);
    }, [labelNames]);

    const handlers: Record<keyof typeof componentShortcuts, (event?: KeyboardEvent) => void> = {
        SWITCH_AUTO_ANNOTATE_STANDARD_CONTROLS: (event: KeyboardEvent | undefined) => {
            if (event) event.preventDefault();
            void handleAutoAnnotate();
        },
    };

    const content = (
        <div style={{ minWidth: 280 }}>
            <div style={{ textAlign: 'center' }}>
                <Text>Generate inference for current frame</Text>
                <br /><br />
                <Select
                    value={selectedModelID}
                    onChange={(value: string): void => setSelectedModelID(value)}
                    loading={modelOptionsLoading}
                    disabled={loading || frameIsDeleted}
                    optionLabelProp='label'
                    style={{ width: '100%', marginBottom: 12, textAlign: 'left' }}
                >
                    <Select.Option value='' label='Task active model'>Task active model</Select.Option>
                    {modelOptions.map((model: any): JSX.Element => (
                        <Select.Option key={model.id} value={model.id} label={model.name || model.id}>
                            <div style={{ display: 'grid', gap: 2 }}>
                                <span>{model.name || model.id}</span>
                                <span style={{ fontSize: 11, color: '#8c8c8c' }}>
                                    {[
                                        model.slot_display_name || model.model_slot || '',
                                        model.source_kind === 'registry_active_slot' ? 'active slot' : (
                                            model.source_kind === 'registry_candidate' ? 'registry candidate' : 'legacy'
                                        ),
                                        model.registry_status || '',
                                    ].filter(Boolean).join(' · ')}
                                </span>
                            </div>
                        </Select.Option>
                    ))}
                </Select>
                <Button
                    type='primary'
                    onClick={handleAutoAnnotate}
                    disabled={loading || frameIsDeleted}
                    loading={loading}
                >
                    Generate inference
                </Button>
            </div>

            <div style={{ marginTop: 16, borderTop: '1px solid #f0f0f0', paddingTop: 12 }}>
                <Text strong>Generate annotations (AI Model)</Text>
                <TextArea
                    rows={4}
                    value={modelPrompt}
                    onChange={(event): void => setModelPrompt(event.target.value)}
                    placeholder='Describe what to annotate on the frame'
                    style={{ marginTop: 8, marginBottom: 8 }}
                />
                <Button
                    onClick={handleAutofillPromptWithLabels}
                    disabled={loading || frameIsDeleted}
                    style={{ marginBottom: 8 }}
                    block
                >
                    Autofill prompt with task labels
                </Button>
                <Input
                    value={contextFrameText}
                    onChange={(event): void => setContextFrameText(event.target.value)}
                    placeholder='Context frame (optional, single frame number)'
                    style={{ marginBottom: 8 }}
                />
                <Button
                    type='primary'
                    onClick={handleModelAnnotate}
                    disabled={loading || frameIsDeleted || !modelPrompt.trim()}
                    loading={loading}
                    block
                >
                    Generate annotations (AI Model)
                </Button>
            </div>
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
                        className='cvat-auto-annotate-control cvat-canvas-control'
                        type='link'
                        disabled={frameIsDeleted}>
                        <RobotOutlined />
                    </Button>
                </CustomPopover>
            </CVATTooltip>
        </>
    );
}

export default React.memo(AutoAnnotateControlComponent);
