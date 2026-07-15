suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(scales)
  library(tibble)
  library(tidyr)
})

CN_FONT <- "Microsoft YaHei"
NEUTRAL_DARK <- "#30343B"
NEUTRAL_MID <- "#77808F"
NEUTRAL_LIGHT <- "#DDE2E8"
SIGNAL_BLUE <- "#3F7392"
SIGNAL_BLUE_LIGHT <- "#B9D0DE"
SIGNAL_TEAL <- "#2F857C"
SIGNAL_TEAL_LIGHT <- "#D7E9E5"
SPLIT_42 <- "#3F7392"
SPLIT_43 <- "#B57932"
ACCENT <- "#B65C46"
WHITE <- "#FFFFFF"

parse_args <- function(args) {
  values <- list(
    input_dir = "outputs/current_mainline_v2/temporal_independent_v3",
    output_dir = "outputs/current_mainline_v2/temporal_independent_v3/figures"
  )
  index <- 1
  while (index <= length(args)) {
    key <- args[[index]]
    if (key == "--input-dir" && index < length(args)) {
      values$input_dir <- args[[index + 1]]
      index <- index + 2
    } else if (key == "--output-dir" && index < length(args)) {
      values$output_dir <- args[[index + 1]]
      index <- index + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }
  values
}

read_report <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("Required report does not exist: %s", path))
  }
  fromJSON(path, simplifyVector = TRUE)
}

assert_finite <- function(values, label) {
  if (length(values) == 0 || any(!is.finite(as.numeric(values)))) {
    stop(sprintf("%s contains missing or non-finite values.", label))
  }
}

write_source_csv <- function(data, path) {
  connection <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(connection), add = TRUE)
  write.csv(data, connection, row.names = FALSE, na = "")
}

theme_nature_cn <- function(base_size = 6.6) {
  theme_classic(base_size = base_size, base_family = CN_FONT) +
    theme(
      axis.line = element_line(linewidth = 0.34, colour = NEUTRAL_DARK),
      axis.ticks = element_line(linewidth = 0.34, colour = NEUTRAL_DARK),
      axis.ticks.length = grid::unit(1.7, "pt"),
      axis.title = element_text(size = base_size, colour = NEUTRAL_DARK),
      axis.text = element_text(size = base_size - 0.45, colour = NEUTRAL_DARK),
      plot.title = element_text(
        size = base_size + 0.8,
        face = "bold",
        colour = NEUTRAL_DARK,
        margin = margin(b = 1.5)
      ),
      plot.subtitle = element_text(
        size = base_size - 0.45,
        colour = NEUTRAL_MID,
        margin = margin(b = 4)
      ),
      plot.tag = element_text(size = 8.2, face = "bold", colour = NEUTRAL_DARK),
      plot.tag.position = c(0, 1),
      plot.margin = margin(4, 7, 4, 6, unit = "pt"),
      panel.grid = element_blank(),
      legend.position = "none"
    )
}

save_figure <- function(plot, output_base, width_mm = 183, height_mm = 125) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4

  svglite::svglite(
    paste0(output_base, ".svg"),
    width = width_in,
    height = height_in,
    bg = WHITE
  )
  print(plot)
  grDevices::dev.off()

  grDevices::cairo_pdf(
    paste0(output_base, ".pdf"),
    width = width_in,
    height = height_in,
    family = CN_FONT,
    onefile = TRUE,
    bg = WHITE
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_tiff(
    paste0(output_base, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 600,
    background = WHITE,
    compression = "lzw"
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(
    paste0(output_base, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 300,
    background = WHITE
  )
  print(plot)
  grDevices::dev.off()
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  input_dir <- args$input_dir
  output_dir <- args$output_dir
  source_dir <- file.path(output_dir, "figure_source_data")
  dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

  consensus <- read_report(file.path(
    input_dir,
    "cross_split_consensus",
    "cross_split_consensus_summary.json"
  ))
  ablation <- read_report(file.path(input_dir, "ablation", "ablation_summary.json"))
  seed42_report <- read_report(file.path(input_dir, "split42_seed_sweep_manifest.json"))
  seed43_report <- read_report(file.path(input_dir, "split43_seed_sweep_manifest.json"))

  if (!identical(consensus$selection_uses_test_labels, FALSE)) {
    stop("Consensus selection must not use test labels.")
  }
  if (!isTRUE(consensus$selection$selected$meets_all_split_retention)) {
    stop("Selected consensus alpha does not meet the stated cross-split retention rule.")
  }

  expected_seeds <- as.integer(c(7, 21, 42, 123, 2026))
  if (!identical(as.integer(seed42_report$seeds), expected_seeds) ||
      !identical(as.integer(seed43_report$seeds), expected_seeds)) {
    stop("Seed manifests do not match the expected five-seed protocol.")
  }

  aggregate <- consensus$aggregate
  selected_alpha <- as.numeric(consensus$selection$selected$alpha)
  split_rows <- as_tibble(consensus$splits) %>%
    transmute(
      split_seed = as.integer(split_seed),
      split = sprintf("split %d", split_seed),
      reference_c_index = as.numeric(test_reference_c_index),
      selected_c_index = as.numeric(test_selected_c_index),
      delta_c_index = as.numeric(test_c_index_delta),
      reference_cox_loss = as.numeric(test_reference_calibrated_cox_loss),
      selected_cox_loss = as.numeric(test_selected_calibrated_cox_loss),
      delta_cox_loss = as.numeric(test_calibrated_cox_loss_delta)
    ) %>%
    arrange(split_seed)

  stage_data <- tibble(
    stage_id = 1:5,
    stage = c(
      "当前主线",
      "AFT +\n旧摘要",
      "AFT +\n边身份",
      "完整\n时间-拓扑",
      "5-seed\n稳健共识"
    ),
    protocol = c(
      "current_mainline_cross_split_mean",
      "seed7_ablation_cross_split_mean",
      "seed7_ablation_cross_split_mean",
      "seed7_ablation_cross_split_mean",
      "validation_only_cross_split_consensus"
    ),
    c_index = c(
      as.numeric(aggregate$mean_reference_test_c_index),
      as.numeric(ablation$aggregate$legacy_summary$mean_selected_test_c_index),
      as.numeric(ablation$aggregate$edge_identity$mean_selected_test_c_index),
      as.numeric(ablation$aggregate$full$mean_selected_test_c_index),
      as.numeric(aggregate$mean_selected_test_c_index)
    ),
    stage_class = factor(
      c("主线", "旧摘要", "边身份", "完整模型", "稳健共识"),
      levels = c("主线", "旧摘要", "边身份", "完整模型", "稳健共识")
    )
  ) %>%
    mutate(value_label = sprintf("%.4f", c_index))

  split_data <- split_rows %>%
    mutate(
      y = rev(seq_len(n())),
      delta_label = sprintf("%+.4f", delta_c_index),
      reference_label = sprintf("主线 %.4f", reference_c_index),
      selected_label = sprintf("新模型 %.4f", selected_c_index)
    )

  seed_data <- bind_rows(
    as_tibble(seed42_report$runs) %>% mutate(split = "split 42"),
    as_tibble(seed43_report$runs) %>% mutate(split = "split 43")
  ) %>%
    transmute(
      split,
      split_seed = as.integer(split_seed),
      model_seed = as.integer(model_seed),
      seed_index = match(model_seed, expected_seeds),
      selected_c_index = as.numeric(selected_test_c_index),
      reference_c_index = as.numeric(reference_test_c_index),
      delta_c_index = as.numeric(test_c_index_delta)
    ) %>%
    mutate(split = factor(split, levels = c("split 42", "split 43")))

  seed_means <- seed_data %>%
    group_by(split) %>%
    summarise(
      mean_c_index = mean(selected_c_index),
      min_c_index = min(selected_c_index),
      max_c_index = max(selected_c_index),
      .groups = "drop"
    ) %>%
    mutate(mean_label = sprintf("%s 均值 %.4f", split, mean_c_index))

  dual_metric_data <- split_rows %>%
    select(split, delta_c_index, delta_cox_loss) %>%
    bind_rows(tibble(
      split = "跨 split 均值",
      delta_c_index = as.numeric(aggregate$mean_test_c_index_delta),
      delta_cox_loss = as.numeric(aggregate$mean_test_calibrated_cox_loss_delta)
    )) %>%
    mutate(
      split = factor(split, levels = c("split 42", "split 43", "跨 split 均值")),
      point_class = if_else(split == "跨 split 均值", "均值", "split")
    )

  assert_finite(stage_data$c_index, "Panel a C-index")
  assert_finite(split_data$selected_c_index, "Panel b C-index")
  assert_finite(seed_data$selected_c_index, "Panel c C-index")
  assert_finite(dual_metric_data$delta_c_index, "Panel d C-index delta")
  assert_finite(dual_metric_data$delta_cox_loss, "Panel d Cox-loss delta")

  write_source_csv(stage_data, file.path(source_dir, "panel_a_model_evolution.csv"))
  write_source_csv(split_data, file.path(source_dir, "panel_b_cross_split_reproduction.csv"))
  write_source_csv(seed_data, file.path(source_dir, "panel_c_five_seed_stability.csv"))
  write_source_csv(dual_metric_data, file.path(source_dir, "panel_d_dual_metric_gain.csv"))

  metadata <- tibble(
    field = c(
      "dataset_version",
      "dataset_status",
      "split_seeds",
      "model_seeds",
      "consensus_alpha",
      "alpha_selection",
      "selection_uses_test_labels",
      "mean_reference_test_c_index",
      "mean_selected_test_c_index",
      "mean_test_c_index_delta",
      "mean_test_calibrated_cox_loss_delta"
    ),
    value = c(
      "topology_v6",
      "synthetic/noisy augmented",
      "42, 43",
      paste(expected_seeds, collapse = ", "),
      sprintf("%.2f", selected_alpha),
      "validation-only cross-split gain-retention consensus",
      "false",
      sprintf("%.12f", aggregate$mean_reference_test_c_index),
      sprintf("%.12f", aggregate$mean_selected_test_c_index),
      sprintf("%.12f", aggregate$mean_test_c_index_delta),
      sprintf("%.12f", aggregate$mean_test_calibrated_cox_loss_delta)
    )
  )
  write_source_csv(metadata, file.path(source_dir, "figure_metadata.csv"))

  y_grid_a <- seq(0.740, 0.760, by = 0.005)
  panel_a <- ggplot(stage_data, aes(x = stage_id, y = c_index)) +
    geom_hline(
      yintercept = y_grid_a,
      linewidth = 0.24,
      colour = NEUTRAL_LIGHT
    ) +
    geom_line(aes(group = 1), linewidth = 0.75, colour = NEUTRAL_MID) +
    geom_point(
      aes(fill = stage_class),
      shape = 21,
      size = 2.75,
      stroke = 0.55,
      colour = WHITE
    ) +
    geom_text(
      aes(label = value_label),
      nudge_y = 0.00125,
      family = CN_FONT,
      size = 2.1,
      fontface = "bold",
      colour = NEUTRAL_DARK
    ) +
    annotate(
      "segment",
      x = 1,
      xend = 5,
      y = 0.7630,
      yend = 0.7630,
      linewidth = 0.38,
      colour = SIGNAL_TEAL
    ) +
    annotate(
      "segment",
      x = c(1, 5),
      xend = c(1, 5),
      y = 0.76255,
      yend = 0.7630,
      linewidth = 0.38,
      colour = SIGNAL_TEAL
    ) +
    annotate(
      "text",
      x = 3,
      y = 0.76335,
      label = sprintf("总体 %+.4f", aggregate$mean_test_c_index_delta),
      family = CN_FONT,
      size = 2.0,
      fontface = "bold",
      colour = SIGNAL_TEAL
    ) +
    annotate(
      "text",
      x = 1.02,
      y = 0.76065,
      label = "历史探索潜力 0.8967（未复现，不纳入纵轴）",
      family = CN_FONT,
      size = 1.75,
      hjust = 0,
      colour = SPLIT_43
    ) +
    scale_fill_manual(values = c(
      "主线" = NEUTRAL_MID,
      "旧摘要" = SIGNAL_BLUE_LIGHT,
      "边身份" = SIGNAL_BLUE,
      "完整模型" = SIGNAL_BLUE,
      "稳健共识" = SIGNAL_TEAL
    )) +
    scale_x_continuous(
      breaks = stage_data$stage_id,
      labels = stage_data$stage,
      limits = c(0.72, 5.28),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = y_grid_a,
      labels = label_number(accuracy = 0.001),
      limits = c(0.7382, 0.7642),
      expand = c(0, 0)
    ) +
    labs(
      tag = "a",
      title = "结构信息驱动主要增益",
      subtitle = "边身份带来跃升，五 seed 共识保留稳健性能",
      x = NULL,
      y = "测试 C-index"
    ) +
    theme_nature_cn() +
    theme(
      axis.line.x = element_line(colour = NEUTRAL_DARK),
      axis.line.y = element_line(colour = NEUTRAL_DARK),
      axis.text.x = element_text(size = 5.6, lineheight = 0.9, margin = margin(t = 3))
    )

  panel_b <- ggplot(split_data, aes(y = y)) +
    geom_segment(
      aes(x = reference_c_index, xend = selected_c_index, yend = y),
      linewidth = 2.5,
      colour = SIGNAL_TEAL_LIGHT,
      lineend = "round"
    ) +
    geom_point(
      aes(x = reference_c_index),
      shape = 21,
      size = 2.6,
      fill = NEUTRAL_MID,
      colour = WHITE,
      stroke = 0.5
    ) +
    geom_point(
      aes(x = selected_c_index),
      shape = 21,
      size = 2.9,
      fill = SIGNAL_TEAL,
      colour = WHITE,
      stroke = 0.55
    ) +
    geom_text(
      aes(x = (reference_c_index + selected_c_index) / 2, y = y + 0.22, label = delta_label),
      family = CN_FONT,
      size = 2.15,
      fontface = "bold",
      colour = SIGNAL_TEAL
    ) +
    geom_text(
      aes(x = reference_c_index, y = y - 0.20, label = reference_label),
      family = CN_FONT,
      size = 1.85,
      colour = NEUTRAL_MID
    ) +
    geom_text(
      aes(x = selected_c_index, y = y - 0.20, label = selected_label),
      family = CN_FONT,
      size = 1.85,
      fontface = "bold",
      colour = SIGNAL_TEAL
    ) +
    scale_x_continuous(
      breaks = c(0.735, 0.745, 0.755, 0.765),
      labels = label_number(accuracy = 0.001),
      limits = c(0.7325, 0.7665),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = split_data$y,
      labels = split_data$split,
      limits = c(0.54, 2.48),
      expand = c(0, 0)
    ) +
    labs(
      tag = "b",
      title = "双 split 独立复现",
      subtitle = sprintf("α = %.2f，仅由 validation 统一选择", selected_alpha),
      x = "测试 C-index",
      y = NULL
    ) +
    theme_nature_cn() +
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())

  split_colours <- c("split 42" = SPLIT_42, "split 43" = SPLIT_43)
  panel_c <- ggplot(seed_data, aes(x = seed_index, y = selected_c_index, colour = split)) +
    geom_hline(
      data = seed_means,
      aes(yintercept = mean_c_index, colour = split),
      linewidth = 0.42,
      linetype = "22",
      alpha = 0.7
    ) +
    geom_line(linewidth = 0.72) +
    geom_point(
      aes(fill = split),
      shape = 21,
      size = 2.35,
      colour = WHITE,
      stroke = 0.48
    ) +
    geom_text(
      data = seed_means,
      aes(x = 5.18, y = mean_c_index, label = mean_label, colour = split),
      family = CN_FONT,
      size = 1.95,
      fontface = "bold",
      hjust = 0,
      inherit.aes = FALSE
    ) +
    annotate(
      "text",
      x = 1,
      y = 0.76225,
      label = "10/10 次运行高于各自主线",
      family = CN_FONT,
      size = 1.9,
      hjust = 0,
      colour = NEUTRAL_MID
    ) +
    scale_colour_manual(values = split_colours) +
    scale_fill_manual(values = split_colours) +
    scale_x_continuous(
      breaks = seq_along(expected_seeds),
      labels = expected_seeds,
      limits = c(0.75, 6.05),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = c(0.753, 0.756, 0.759, 0.762),
      labels = label_number(accuracy = 0.001),
      limits = c(0.7524, 0.7627),
      expand = c(0, 0)
    ) +
    labs(
      tag = "c",
      title = "五 seed 表现稳定",
      subtitle = "每个 seed 均仅以 validation 选参",
      x = "模型 seed",
      y = "测试 C-index"
    ) +
    theme_nature_cn()

  label_positions <- tibble(
    split = factor(c("split 42", "split 43", "跨 split 均值"),
      levels = levels(dual_metric_data$split)
    ),
    label_x = c(0.01815, 0.01555, 0.01705),
    label_y = c(-0.01535, -0.01115, -0.01275),
    hjust = c(0, 1, 0)
  )
  dual_metric_plot_data <- dual_metric_data %>%
    left_join(label_positions, by = "split")
  panel_d <- ggplot(dual_metric_plot_data, aes(x = delta_c_index, y = delta_cox_loss)) +
    annotate(
      "rect",
      xmin = 0,
      xmax = Inf,
      ymin = -Inf,
      ymax = 0,
      fill = SIGNAL_TEAL_LIGHT,
      alpha = 0.52
    ) +
    geom_vline(xintercept = 0, linewidth = 0.38, linetype = "22", colour = NEUTRAL_MID) +
    geom_hline(yintercept = 0, linewidth = 0.38, linetype = "22", colour = NEUTRAL_MID) +
    geom_segment(
      data = filter(dual_metric_plot_data, split != "跨 split 均值"),
      aes(x = 0, y = 0, xend = delta_c_index, yend = delta_cox_loss, colour = split),
      linewidth = 0.58,
      arrow = grid::arrow(length = grid::unit(3, "pt"), type = "closed"),
      inherit.aes = FALSE
    ) +
    geom_point(
      data = filter(dual_metric_plot_data, split != "跨 split 均值"),
      aes(fill = split),
      shape = 21,
      size = 2.55,
      colour = WHITE,
      stroke = 0.5
    ) +
    geom_point(
      data = filter(dual_metric_plot_data, split == "跨 split 均值"),
      shape = 23,
      size = 3.1,
      fill = ACCENT,
      colour = WHITE,
      stroke = 0.55
    ) +
    geom_text(
      aes(x = label_x, y = label_y, label = split, hjust = hjust, colour = split),
      family = CN_FONT,
      size = 1.95,
      fontface = "bold",
      show.legend = FALSE
    ) +
    annotate(
      "text",
      x = 0.0012,
      y = -0.0014,
      label = "双改善区",
      family = CN_FONT,
      size = 1.9,
      fontface = "bold",
      hjust = 0,
      colour = SIGNAL_TEAL
    ) +
    scale_colour_manual(values = c(split_colours, "跨 split 均值" = ACCENT)) +
    scale_fill_manual(values = split_colours) +
    scale_x_continuous(
      breaks = c(0, 0.005, 0.010, 0.015, 0.020),
      labels = label_number(accuracy = 0.001),
      limits = c(-0.0006, 0.0210),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = c(-0.015, -0.010, -0.005, 0),
      labels = label_number(accuracy = 0.001),
      limits = c(-0.0164, 0.0011),
      expand = c(0, 0)
    ) +
    labs(
      tag = "d",
      title = "排序与损失同步改善",
      subtitle = "右下象限表示两项指标同时变好",
      x = "Δ C-index（越大越好）",
      y = "Δ calibrated Cox loss（越小越好）"
    ) +
    theme_nature_cn()

  figure <- ((panel_a | panel_b) / (panel_c | panel_d)) +
    plot_layout(widths = c(1.06, 1), heights = c(1, 1.03)) +
    plot_annotation(
      title = "时间-拓扑生存模型的性能提升与复现性",
      subtitle = sprintf(
        "平均测试 C-index %.4f → %.4f（%+.4f）；0.8967 保留为历史探索潜力（未复现）",
        aggregate$mean_reference_test_c_index,
        aggregate$mean_selected_test_c_index,
        aggregate$mean_test_c_index_delta
      ),
      caption = paste0(
        "split seeds = 42, 43；model seeds = 7, 21, 42, 123, 2026；",
        sprintf("共识 α = %.2f，仅由 validation 选择。", selected_alpha),
        "两个 split 的校准 Cox loss 均下降；数据：topology_v6（synthetic/noisy augmented）。"
      ),
      theme = theme(
        text = element_text(family = CN_FONT, colour = NEUTRAL_DARK),
        plot.title = element_text(size = 11.2, face = "bold", margin = margin(b = 2)),
        plot.subtitle = element_text(size = 6.7, colour = NEUTRAL_MID, margin = margin(b = 7)),
        plot.caption = element_text(
          size = 5.35,
          colour = NEUTRAL_MID,
          hjust = 0,
          margin = margin(t = 6)
        ),
        plot.margin = margin(7, 7, 5, 7, unit = "pt")
      )
    )

  output_base <- file.path(output_dir, "fig_temporal_topology_reproducibility_v3")
  save_figure(figure, output_base)

  qa_notes <- c(
    "Figure contract",
    "Core conclusion: The censor-aware temporal-topology model reproducibly improves test C-index across two splits and five model seeds; validation-only consensus also lowers calibrated Cox loss.",
    "Archetype: quantitative 2 x 2 grid.",
    "Backend: R only (ggplot2 + patchwork; svglite/cairo_pdf/ragg export).",
    "Final size: 183 x 125 mm.",
    "Panel a: model-information ablation and robust consensus.",
    "Panel b: independent split-level baseline-to-consensus change.",
    "Panel c: five-seed stability for both splits.",
    "Panel d: joint C-index and calibrated Cox-loss deltas.",
    "Statistics: two split seeds (42, 43); five model seeds per split (7, 21, 42, 123, 2026).",
    "Selection: alpha = 0.63, selected by validation-only cross-split gain retention; test labels were not used.",
    "Uncertainty: n = 2 split seeds is insufficient for a formal confidence interval; the figure therefore shows exact split and seed results without inferential p-values.",
    "Baseline: current mainline predictions paired within each split.",
    "Dataset limitation: topology_v6 is synthetic/noisy augmented and is not an external clinical validation cohort.",
    "Historical potential: C-index 0.8967 is retained as an exploratory single-run upper-bound signal; it is not included in the strict new-model axis or inference.",
    "Source data: figure_source_data/*.csv."
  )
  writeLines(qa_notes, file.path(output_dir, "figure_qa_notes.txt"), useBytes = TRUE)

  message(sprintf("Saved figure bundle to %s", normalizePath(output_dir, winslash = "/")))
  message(sprintf(
    "Mean test C-index: %.6f -> %.6f (%+.6f)",
    aggregate$mean_reference_test_c_index,
    aggregate$mean_selected_test_c_index,
    aggregate$mean_test_c_index_delta
  ))
}

main()
