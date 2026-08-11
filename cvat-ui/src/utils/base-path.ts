// Copyright (C) 2026 LuxTronic
//
// SPDX-License-Identifier: MIT

const rawBasePath = (process.env.CVAT_UI_BASE_PATH ?? '').trim();

function normalizeBasePath(value: string): string {
    if (!value || value === '/') {
        return '';
    }

    const trimmed = value.replace(/^\/+|\/+$/g, '');
    return trimmed ? `/${trimmed}` : '';
}

const UI_BASE_PATH = normalizeBasePath(rawBasePath);

export function getUIBasePath(): string {
    return UI_BASE_PATH;
}

export function withUIBasePath(path = ''): string {
    if (!path) {
        return UI_BASE_PATH || '/';
    }

    const absolutePath = path.startsWith('/') ? path : `/${path}`;
    return `${UI_BASE_PATH}${absolutePath}` || '/';
}
