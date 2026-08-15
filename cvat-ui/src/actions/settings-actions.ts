// Copyright (C) 2020-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import _ from 'lodash';
import { AnyAction } from 'redux';
import { ThunkAction } from 'utils/redux';
import {
    GridColor, ColorBy, SettingsState, ToolsBlockerState,
    CombinedState,
} from 'reducers';
import { OrientationVisibility } from 'cvat-canvas3d-wrapper';
import { SerializedImageFilter } from 'cvat-core-wrapper';
import { ImageFilter, ImageFilterAlias } from 'utils/image-processing';
import GammaCorrection, { GammaFilterOptions } from 'utils/fabric-wrapper/gamma-correction';
import { resolveConflicts } from 'utils/conflict-detector';
import { shortcutsActions } from './shortcuts-actions';

// Version of the persisted `clientSettings` blob.
//
// Bumped when a default changes in a way that must reach browsers which already
// have settings stored, since a stored value always wins over a new default.
// Settings saved before versioning are treated as version 0.
//
//   1 - autoSave defaults to true (previously false). Unversioned settings are
//       treated as the legacy default and migrated to true; an explicit opt-out
//       made after this version is tracked by autoSavePreferenceSet.
const CLIENT_SETTINGS_VERSION = 1;

// Workspace keys that a given migration must not restore from storage, so the
// new default survives the restore for exactly one load. After that the blob is
// rewritten with the current version and the annotator's own choice is honoured.
const MIGRATED_WORKSPACE_KEYS: Record<number, string[]> = {
    1: ['autoSave'],
};

function workspaceKeysToSkip(storedVersion: number, autoSavePreferenceSet: boolean): Set<string> {
    const skip = new Set<string>();
    Object.entries(MIGRATED_WORKSPACE_KEYS).forEach(([version, keys]) => {
        if (storedVersion < Number(version) && !(version === '1' && autoSavePreferenceSet)) {
            keys.forEach((key) => skip.add(key));
        }
    });
    return skip;
}

export enum SettingsActionTypes {
    SWITCH_ROTATE_ALL = 'SWITCH_ROTATE_ALL',
    SWITCH_GRID = 'SWITCH_GRID',
    CHANGE_GRID_SIZE = 'CHANGE_GRID_SIZE',
    CHANGE_GRID_COLOR = 'CHANGE_GRID_COLOR',
    CHANGE_GRID_OPACITY = 'CHANGE_GRID_OPACITY',
    CHANGE_SHAPES_OPACITY = 'CHANGE_SHAPES_OPACITY',
    CHANGE_SELECTED_SHAPES_OPACITY = 'CHANGE_SELECTED_SHAPES_OPACITY',
    CHANGE_SHAPES_COLOR_BY = 'CHANGE_SHAPES_COLOR_BY',
    CHANGE_SHAPES_OUTLINED_BORDERS = 'CHANGE_SHAPES_OUTLINED_BORDERS',
    CHANGE_SHAPES_SHOW_PROJECTIONS = 'CHANGE_SHAPES_SHOW_PROJECTIONS',
    CHANGE_SHOW_UNLABELED_REGIONS = 'CHANGE_SHOW_UNLABELED_REGIONS',
    CHANGE_SHOW_GROUND_TRUTH = 'CHANGE_SHOW_GROUND_TRUTH',
    CHANGE_FRAME_STEP = 'CHANGE_FRAME_STEP',
    CHANGE_FRAME_SPEED = 'CHANGE_FRAME_SPEED',
    SWITCH_RESET_ZOOM = 'SWITCH_RESET_ZOOM',
    SWITCH_SMOOTH_IMAGE = 'SWITCH_SMOOTH_IMAGE',
    SWITCH_TEXT_FONT_SIZE = 'SWITCH_TEXT_FONT_SIZE',
    SWITCH_CONTROL_POINTS_SIZE = 'SWITCH_CONTROL_POINTS_SIZE',
    SWITCH_TEXT_POSITION = 'SWITCH_TEXT_POSITION',
    SWITCH_TEXT_CONTENT = 'SWITCH_TEXT_CONTENT',
    CHANGE_BRIGHTNESS_LEVEL = 'CHANGE_BRIGHTNESS_LEVEL',
    CHANGE_CONTRAST_LEVEL = 'CHANGE_CONTRAST_LEVEL',
    CHANGE_SATURATION_LEVEL = 'CHANGE_SATURATION_LEVEL',
    SWITCH_AUTO_SAVE = 'SWITCH_AUTO_SAVE',
    CHANGE_AUTO_SAVE_INTERVAL = 'CHANGE_AUTO_SAVE_INTERVAL',
    CHANGE_FOCUSED_OBJECT_PADDING = 'CHANGE_FOCUSED_OBJECT_PADDING',
    CHANGE_DEFAULT_APPROX_POLY_THRESHOLD = 'CHANGE_DEFAULT_APPROX_POLY_THRESHOLD',
    SWITCH_AUTOMATIC_BORDERING = 'SWITCH_AUTOMATIC_BORDERING',
    SWITCH_SNAP_TO_POINT = 'SWITCH_SNAP_TO_POINT',
    SWITCH_ADAPTIVE_ZOOM = 'SWITCH_ADAPTIVE_ZOOM',
    SWITCH_INTELLIGENT_POLYGON_CROP = 'SWITCH_INTELLIGENT_POLYGON_CROP',
    SWITCH_SHOWNIG_INTERPOLATED_TRACKS = 'SWITCH_SHOWNIG_INTERPOLATED_TRACKS',
    SWITCH_SHOWING_OBJECTS_TEXT_ALWAYS = 'SWITCH_SHOWING_OBJECTS_TEXT_ALWAYS',
    CHANGE_CANVAS_BACKGROUND_COLOR = 'CHANGE_CANVAS_BACKGROUND_COLOR',
    SWITCH_SETTINGS_DIALOG = 'SWITCH_SETTINGS_DIALOG',
    SET_SETTINGS = 'SET_SETTINGS',
    SWITCH_TOOLS_BLOCKER_STATE = 'SWITCH_TOOLS_BLOCKER_STATE',
    SWITCH_SHOWING_DELETED_FRAMES = 'SWITCH_SHOWING_DELETED_FRAMES',
    SWITCH_SHOWING_TAGS_ON_FRAME = 'SWITCH_SHOWING_TAGS_ON_FRAME',
    ENABLE_IMAGE_FILTER = 'ENABLE_IMAGE_FILTER',
    DISABLE_IMAGE_FILTER = 'DISABLE_IMAGE_FILTER',
    RESET_IMAGE_FILTERS = 'RESET_IMAGE_FILTERS',
    CHANGE_SHAPES_ORIENTATION_VISIBILITY = 'CHANGE_SHAPES_ORIENTATION_VISIBILITY',
}

export function changeShapesOpacity(opacity: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_OPACITY,
        payload: {
            opacity,
        },
    };
}

export function changeSelectedShapesOpacity(selectedOpacity: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SELECTED_SHAPES_OPACITY,
        payload: {
            selectedOpacity,
        },
    };
}

export function changeShapesColorBy(colorBy: ColorBy): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_COLOR_BY,
        payload: {
            colorBy,
        },
    };
}

export function changeShowGroundTruth(showGroundTruth: boolean): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHOW_GROUND_TRUTH,
        payload: {
            showGroundTruth,
        },
    };
}

export function changeShapesOutlinedBorders(outlined: boolean, color: string): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_OUTLINED_BORDERS,
        payload: {
            outlined,
            color,
        },
    };
}

export function changeShowBitmap(showBitmap: boolean): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHOW_UNLABELED_REGIONS,
        payload: {
            showBitmap,
        },
    };
}

export function changeShowProjections(showProjections: boolean): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_SHOW_PROJECTIONS,
        payload: {
            showProjections,
        },
    };
}

export function changeOrientationVisibility(orientationVisibility: Partial<OrientationVisibility>): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_ORIENTATION_VISIBILITY,
        payload: {
            orientationVisibility,
        },
    };
}

export function switchRotateAll(rotateAll: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_ROTATE_ALL,
        payload: {
            rotateAll,
        },
    };
}

export function switchGrid(grid: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_GRID,
        payload: {
            grid,
        },
    };
}

export function changeGridSize(gridSize: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_GRID_SIZE,
        payload: {
            gridSize,
        },
    };
}

export function changeGridColor(gridColor: GridColor): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_GRID_COLOR,
        payload: {
            gridColor,
        },
    };
}

export function changeGridOpacity(gridOpacity: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_GRID_OPACITY,
        payload: {
            gridOpacity,
        },
    };
}

export function changeFrameStep(frameStep: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_FRAME_STEP,
        payload: {
            frameStep,
        },
    };
}

export function changeFrameSpeed(frameSpeed: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_FRAME_SPEED,
        payload: {
            frameSpeed,
        },
    };
}

export function switchResetZoom(resetZoom: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_RESET_ZOOM,
        payload: {
            resetZoom,
        },
    };
}

export function switchSmoothImage(enabled: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SMOOTH_IMAGE,
        payload: {
            smoothImage: enabled,
        },
    };
}

export function switchTextFontSize(fontSize: number): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_TEXT_FONT_SIZE,
        payload: {
            fontSize,
        },
    };
}

export function switchControlPointsSize(pointsSize: number): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_CONTROL_POINTS_SIZE,
        payload: {
            controlPointsSize: pointsSize,
        },
    };
}

export function switchTextPosition(position: 'auto' | 'center'): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_TEXT_POSITION,
        payload: {
            position,
        },
    };
}

export function switchTextContent(textContent: string[]): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_TEXT_CONTENT,
        payload: {
            textContent: textContent.join(','),
        },
    };
}

export function changeBrightnessLevel(level: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_BRIGHTNESS_LEVEL,
        payload: {
            level,
        },
    };
}

export function changeContrastLevel(level: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_CONTRAST_LEVEL,
        payload: {
            level,
        },
    };
}

export function changeSaturationLevel(level: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SATURATION_LEVEL,
        payload: {
            level,
        },
    };
}

export function switchAutoSave(autoSave: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_AUTO_SAVE,
        payload: {
            autoSave,
        },
    };
}

export function changeAutoSaveInterval(autoSaveInterval: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_AUTO_SAVE_INTERVAL,
        payload: {
            autoSaveInterval,
        },
    };
}

export function changeFocusedObjectPadding(focusedObjectPadding: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_FOCUSED_OBJECT_PADDING,
        payload: {
            focusedObjectPadding,
        },
    };
}

export function switchShowingInterpolatedTracks(showAllInterpolationTracks: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SHOWNIG_INTERPOLATED_TRACKS,
        payload: {
            showAllInterpolationTracks,
        },
    };
}

export function switchShowingObjectsTextAlways(showObjectsTextAlways: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SHOWING_OBJECTS_TEXT_ALWAYS,
        payload: {
            showObjectsTextAlways,
        },
    };
}

export function switchAutomaticBordering(automaticBordering: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_AUTOMATIC_BORDERING,
        payload: {
            automaticBordering,
        },
    };
}

export function switchSnapToPoint(snapToPoint: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SNAP_TO_POINT,
        payload: {
            snapToPoint,
        },
    };
}

export function switchAdaptiveZoom(adaptiveZoom: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_ADAPTIVE_ZOOM,
        payload: {
            adaptiveZoom,
        },
    };
}

export function switchIntelligentPolygonCrop(intelligentPolygonCrop: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_INTELLIGENT_POLYGON_CROP,
        payload: {
            intelligentPolygonCrop,
        },
    };
}

export function changeCanvasBackgroundColor(color: string): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_CANVAS_BACKGROUND_COLOR,
        payload: {
            color,
        },
    };
}

export function switchSettingsModalVisible(visible: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SETTINGS_DIALOG,
        payload: { visible },
    };
}

export function changeDefaultApproxPolyAccuracy(approxPolyAccuracy: number): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_DEFAULT_APPROX_POLY_THRESHOLD,
        payload: {
            approxPolyAccuracy,
        },
    };
}

export function switchToolsBlockerState(toolsBlockerState: ToolsBlockerState): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_TOOLS_BLOCKER_STATE,
        payload: {
            toolsBlockerState,
        },
    };
}

export function setSettings(settings: Partial<SettingsState>): AnyAction {
    return {
        type: SettingsActionTypes.SET_SETTINGS,
        payload: {
            settings,
        },
    };
}

export function switchShowingDeletedFrames(showDeletedFrames: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SHOWING_DELETED_FRAMES,
        payload: {
            showDeletedFrames,
        },
    };
}

export function switchShowingTagsOnFrame(showTagsOnFrame: boolean): AnyAction {
    return {
        type: SettingsActionTypes.SWITCH_SHOWING_TAGS_ON_FRAME,
        payload: {
            showTagsOnFrame,
        },
    };
}

export function enableImageFilter(filter: ImageFilter, options: object | null = null): AnyAction {
    return {
        type: SettingsActionTypes.ENABLE_IMAGE_FILTER,
        payload: {
            filter,
            options,
        },
    };
}

export function disableImageFilter(filterAlias: ImageFilterAlias): AnyAction {
    return {
        type: SettingsActionTypes.DISABLE_IMAGE_FILTER,
        payload: {
            filterAlias,
        },
    };
}

export function resetImageFilters(): AnyAction {
    return {
        type: SettingsActionTypes.RESET_IMAGE_FILTERS,
        payload: {},
    };
}

export function restoreSettingsAsync(): ThunkAction {
    return async (dispatch, getState): Promise<void> => {
        const state: CombinedState = getState();
        const { settings, shortcuts } = state;

        dispatch(shortcutsActions.setDefaultShortcuts(structuredClone(shortcuts.keyMap)));

        const settingsString = localStorage.getItem('clientSettings') as string;
        if (!settingsString) return;

        const loadedSettings = JSON.parse(settingsString);
        const newSettings = {
            player: settings.player,
            workspace: settings.workspace,
            imageFilters: [],
        } as Pick<SettingsState, 'player' | 'workspace' | 'imageFilters'>;

        const storedVersion = Number(loadedSettings.version) || 0;
        const skipWorkspaceKeys = workspaceKeysToSkip(
            storedVersion,
            loadedSettings.workspace?.autoSavePreferenceSet === true,
        );

        Object.entries(_.pick(newSettings, ['player', 'workspace'])).forEach(([sectionKey, section]) => {
            Object.keys(section).forEach((key) => {
                // A migrated key is left at its new default for this load rather than
                // restored. Skipping it here matters: the defineProperty below writes a
                // non-writable property, so the value cannot be corrected afterwards.
                if (sectionKey === 'workspace' && skipWorkspaceKeys.has(key)) {
                    return;
                }

                const setValue = loadedSettings[sectionKey]?.[key];
                if (setValue !== undefined) {
                    Object.defineProperty(newSettings[sectionKey as 'player' | 'workspace'], key, { value: setValue });
                }
            });
        });

        if ('imageFilters' in loadedSettings) {
            loadedSettings.imageFilters.forEach((filter: SerializedImageFilter) => {
                if (filter.alias === ImageFilterAlias.GAMMA_CORRECTION) {
                    newSettings.imageFilters.push({
                        modifier: new GammaCorrection(filter.params as GammaFilterOptions),
                        alias: ImageFilterAlias.GAMMA_CORRECTION,
                    });
                }
            });
        }

        dispatch(setSettings(newSettings));

        if ('shortcuts' in loadedSettings) {
            const updateKeyMap = structuredClone(shortcuts.keyMap);

            Object.entries(loadedSettings.shortcuts.keyMap).forEach(([key, value]) => {
                if (key in updateKeyMap) {
                    updateKeyMap[key].sequences = (value as { sequences: string[] }).sequences;
                }
            });

            const resolvedKeyMap = resolveConflicts(updateKeyMap, shortcuts.keyMap);

            dispatch(shortcutsActions.registerShortcuts(resolvedKeyMap));
        }
    };
}

export function updateCachedSettings(settings: CombinedState['settings'], shortcuts: CombinedState['shortcuts']): void {
    const supportedImageFilters = [ImageFilterAlias.GAMMA_CORRECTION];
    const settingsForSaving = {
        // Stamping the version here is what makes a migration one-shot: this runs
        // immediately after restoreSettingsAsync on load, so the next load sees the
        // current version and stops overriding the migrated key.
        version: CLIENT_SETTINGS_VERSION,
        player: settings.player,
        workspace: settings.workspace,
        shortcuts: {
            keyMap: Object.entries(shortcuts.keyMap).reduce<Record<string, { sequences: string[] }>>(
                (acc, [key, value]) => {
                    if (key in shortcuts.defaultState) {
                        acc[key] = { sequences: value.sequences };
                    }
                    return acc;
                }, {}),
        },
        imageFilters: settings.imageFilters.filter((imageFilter) => supportedImageFilters.includes(imageFilter.alias))
            .map((imageFilter) => imageFilter.modifier.toJSON()),
    };

    localStorage.setItem('clientSettings', JSON.stringify(settingsForSaving));
}
