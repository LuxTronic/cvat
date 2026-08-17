// Copyright (C) 2021-2022 Intel Corporation
// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/// <reference types="cypress" />

import { taskName } from '../../support/const';

context('Settings. "Auto save" option.', () => {
    const caseId = '51';

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
    });
});
