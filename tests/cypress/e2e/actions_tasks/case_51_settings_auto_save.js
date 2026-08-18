// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

import { taskName } from '../../support/const';

context('Settings. "Auto save" option.', () => {
    const caseId = '51';

    // Rewrites the persisted clientSettings blob and reloads, so restoreSettingsAsync runs
    // against it the way it would in an annotator's browser.
    function seedStoredSettings(workspace, version) {
        cy.window().then((win) => {
            const stored = JSON.parse(win.localStorage.getItem('clientSettings') || '{}');
            const next = { ...stored, workspace: { ...(stored.workspace || {}), ...workspace } };
            if (version === null) {
                delete next.version;
            } else {
                next.version = version;
            }
            win.localStorage.setItem('clientSettings', JSON.stringify(next));
        });
        cy.reload();
        cy.get('.cvat-canvas-container').should('exist').and('be.visible');
    }

    function autoSaveCheckbox() {
        cy.openSettings();
        cy.contains('Workspace').click();
        return cy.get('.cvat-workspace-settings-auto-save').find('[type="checkbox"]');
    }

    before(() => {
        cy.prepareUserSession();
        cy.openTaskJob(taskName);
    });

    describe(`Testing case "${caseId}"`, () => {
        it('Auto save is enabled by default and shows the effective interval.', () => {
            cy.openSettings();
            cy.contains('Workspace').click();

            // Asserted before touching the control: this is the regression test for autosave
            // shipping enabled, so the checkbox must be observed in its default state.
            cy.get('.cvat-workspace-settings-auto-save').within(() => {
                cy.get('[type="checkbox"]').should('be.checked');
            });
            cy.get('.cvat-workspace-settings-auto-save-interval')
                .should('contain.text', '15 minutes');

            // Opting out still works, and is restored so the rest of the shard keeps the default.
            cy.get('.cvat-workspace-settings-auto-save').within(() => {
                cy.get('[type="checkbox"]').uncheck();
                cy.get('[type="checkbox"]').should('not.be.checked');
                cy.get('[type="checkbox"]').check();
                cy.get('[type="checkbox"]').should('be.checked');
            });
            cy.closeSettings();
        });

        it('Settings stored before versioning are migrated to autosave enabled.', () => {
            // An unversioned blob cannot distinguish the historical default false from a
            // deliberate opt-out, because nothing recorded the provenance of the choice.
            // The documented policy is to treat it as the old default and enable autosave;
            // this pins that policy so a future change to it has to be deliberate.
            seedStoredSettings({ autoSave: false }, null);

            autoSaveCheckbox().should('be.checked');
            cy.closeSettings();
        });

        it('An opt-out recorded after versioning survives the migration.', () => {
            // autoSavePreferenceSet is what the migration reads to tell a deliberate choice
            // apart from an inherited default, so an annotator who turns autosave off keeps
            // it off across reloads.
            seedStoredSettings({ autoSave: false, autoSavePreferenceSet: true }, 1);

            autoSaveCheckbox().should('not.be.checked');
            cy.closeSettings();

            // Leave the default in place for anything that runs after this spec.
            seedStoredSettings({ autoSave: true, autoSavePreferenceSet: true }, 1);
        });
    });
});
