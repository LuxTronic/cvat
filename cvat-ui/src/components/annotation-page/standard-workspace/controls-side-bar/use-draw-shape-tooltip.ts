// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { useSelector } from 'react-redux';
import { CombinedState } from 'reducers';

// The repeat key is the same for every shape, so the eight draw controls resolve and
// format it here rather than each carrying its own copy. #5 duplicated the string in
// all eight files, which is what made it awkward to reconcile against upstream later.
export default function useDrawShapeTooltip(action: string): string {
    const { normalizedKeyMap } = useSelector((state: CombinedState) => state.shortcuts);
    return `${action} ${normalizedKeyMap.SWITCH_DRAW_MODE_STANDARD_CONTROLS}`;
}
