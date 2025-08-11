# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
_base_ = ['./gaussianflowocc.py']

model = dict(
    render_depth=False,
)