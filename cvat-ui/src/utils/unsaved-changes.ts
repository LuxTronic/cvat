// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { Job, Task } from 'cvat-core-wrapper';

// saveAnnotationsAsync flushes both the annotation collection and the frame meta
// (deleted/restored frames), so "is there anything to save" has to consider both.
// Checking only annotations leaves a lone frame deletion invisible to autosave.
export default function hasUnsavedChanges(instance: Job | Task | null | undefined): boolean {
    if (!instance) {
        return false;
    }

    return instance.annotations.hasUnsavedChanges() || instance.frames.hasUnsavedChanges();
}
