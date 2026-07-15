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
SPLIT_42 <- "#3F7392"
SPLIT_43 <- "#B57932"
RIBBON_42 <- "#C9DBE5"
RIBBON_43 <- "#E8D5BD"
CHANCE <- "#B7A3A3"
WHITE <- "#FFFFFF"

parse_args <- function(args) {
  values <- list(
    input_dir = "outputs/current_mainline_v2/temporal_independent_v3/figure_diagnostics",
    output_dir = "outputs/current_mainline_v2/temporal_independent_v3/figures",
    compact = FALSE
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
    } else if (key == "--compact") {
      values$compact <- TRUE
      index <- index + 1
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }
  values
}

theme_nature_cn <- function(base_size = 6.35) {
  theme_classic(base_size = base_size, base_family = CN_FONT) +
    theme(
      axis.line = element_line(linewidth = 0.33, colour = NEUTRAL_DARK),
      axis.ticks = element_line(linewidth = 0.33, colour = NEUTRAL_DARK),
      axis.ticks.length = grid::unit(1.6, "pt"),
      axis.title = element_text(size = base_size, colour = NEUTRAL_DARK),
      axis.text = element_text(size = base_size - 0.45, colour = NEUTRAL_DARK),
      plot.title = element_text(
        size = base_size + 0.75,
        face = "bold",
        colour = NEUTRAL_DARK,
        margin = margin(b = 1.2)
      ),
      plot.subtitle = element_text(
        size = base_size - 0.55,
        colour = NEUTRAL_MID,
        margin = margin(b = 3.5)
      ),
      plot.tag = element_text(size = 8.1, face = "bold", colour = NEUTRAL_DARK),
      plot.tag.position = c(0, 1),
      plot.margin = margin(4, 6, 4, 5, unit = "pt"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.8),
      legend.key.width = grid::unit(10, "pt"),
      legend.key.height = grid::unit(5, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      panel.grid = element_blank()
    )
}

save_figure <- function(plot, output_base, width_mm = 183, height_mm = 126) {
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

write_source_csv <- function(data, path) {
  connection <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(connection), add = TRUE)
  write.csv(data, connection, row.names = FALSE, na = "")
}

read_roc_report <- function(path, split_label) {
  if (!file.exists(path)) {
    stop(sprintf("Required ROC report does not exist: %s", path))
  }
  report <- fromJSON(path, simplifyVector = FALSE)
  curve_rows <- lapply(report$horizons, function(row) {
    tibble(
      split = split_label,
      horizon = as.numeric(row$horizon),
      auc = as.numeric(row$auc),
      num_cases = as.integer(row$num_cases),
      num_controls = as.integer(row$num_controls),
      num_excluded_censored = as.integer(row$num_excluded_censored),
      fpr = as.numeric(unlist(row$false_positive_rate)),
      tpr = as.numeric(unlist(row$true_positive_rate))
    )
  })
  bind_rows(curve_rows)
}

interpolate_trajectories <- function(long_data, progress_grid = seq(2, 100, by = 2)) {
  long_data %>%
    group_by(split, split_seed, model_seed, selected_expert, metric) %>%
    group_modify(function(data, key) {
      tibble(
        training_progress_pct = progress_grid,
        value = approx(
          x = data$training_progress_pct,
          y = data$value,
          xout = progress_grid,
          method = "linear",
          rule = 2,
          ties = mean
        )$y
      )
    }) %>%
    ungroup()
}

make_training_panel <- function(
    raw_data,
    summary_data,
    metric_name,
    tag,
    title,
    subtitle,
    y_label,
    log_y = FALSE,
    y_limits = NULL,
    y_breaks = NULL) {
  raw_metric <- filter(raw_data, metric == metric_name)
  summary_metric <- filter(summary_data, metric == metric_name)
  panel <- ggplot() +
    geom_ribbon(
      data = summary_metric,
      aes(
        x = training_progress_pct,
        ymin = min_value,
        ymax = max_value,
        fill = split,
        group = split
      ),
      alpha = 0.28,
      colour = NA
    ) +
    geom_line(
      data = raw_metric,
      aes(
        x = training_progress_pct,
        y = value,
        colour = split,
        group = interaction(split, model_seed)
      ),
      linewidth = 0.28,
      alpha = 0.34
    ) +
    geom_line(
      data = summary_metric,
      aes(
        x = training_progress_pct,
        y = median_value,
        colour = split,
        group = split
      ),
      linewidth = 0.82
    ) +
    scale_colour_manual(values = c("split 42" = SPLIT_42, "split 43" = SPLIT_43)) +
    scale_fill_manual(values = c("split 42" = RIBBON_42, "split 43" = RIBBON_43), guide = "none") +
    scale_x_continuous(
      breaks = c(0, 25, 50, 75, 100),
      limits = c(0, 100),
      expand = c(0, 0)
    ) +
    labs(
      tag = tag,
      title = title,
      subtitle = subtitle,
      x = "训练进度（% 最优迭代）",
      y = y_label,
      colour = NULL
    ) +
    theme_nature_cn()
  if (log_y) {
    panel <- panel + scale_y_log10(
      breaks = y_breaks,
      labels = label_number(accuracy = 0.1),
      limits = y_limits
    )
  } else {
    panel <- panel + scale_y_continuous(
      breaks = y_breaks,
      labels = label_number(accuracy = 0.01),
      limits = y_limits,
      expand = expansion(mult = c(0.02, 0.04))
    )
  }
  panel
}

make_roc_panel <- function(roc_data, horizon_value, tag) {
  panel_data <- filter(roc_data, horizon == horizon_value)
  auc_rows <- panel_data %>%
    distinct(split, horizon, auc, num_cases, num_controls, num_excluded_censored) %>%
    arrange(split)
  auc42 <- auc_rows$auc[auc_rows$split == "split 42"]
  auc43 <- auc_rows$auc[auc_rows$split == "split 43"]
  ggplot(panel_data, aes(x = fpr, y = tpr, colour = split)) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linewidth = 0.4,
      linetype = "44",
      colour = CHANCE
    ) +
    geom_step(linewidth = 0.62, direction = "hv") +
    scale_colour_manual(values = c("split 42" = SPLIT_42, "split 43" = SPLIT_43)) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.25),
      labels = label_number(accuracy = 0.01),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.25),
      labels = label_number(accuracy = 0.01),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    coord_equal() +
    labs(
      tag = tag,
      title = sprintf("%.0f 时间单位", horizon_value),
      subtitle = sprintf("AUC %.3f / %.3f", auc42, auc43),
      x = "假阳性率",
      y = "真阳性率",
      colour = NULL
    ) +
    theme_nature_cn() +
    theme(
      axis.line = element_blank(),
      panel.border = element_rect(fill = NA, colour = NEUTRAL_DARK, linewidth = 0.33)
    )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  input_dir <- args$input_dir
  output_dir <- args$output_dir
  source_dir <- file.path(output_dir, "figure_source_data_training_roc")
  dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

  trajectory_path <- file.path(input_dir, "selected_expert_training_trajectories.csv")
  if (!file.exists(trajectory_path)) {
    stop(sprintf("Required training trajectory data does not exist: %s", trajectory_path))
  }
  trajectory <- read.csv(trajectory_path, stringsAsFactors = FALSE, check.names = FALSE) %>%
    mutate(
      split = factor(sprintf("split %d", split_seed), levels = c("split 42", "split 43")),
      model_seed = factor(model_seed, levels = c(7, 21, 42, 123, 2026))
    )
  required_columns <- c(
    "split_seed",
    "model_seed",
    "selected_expert",
    "training_progress_pct",
    "train_aft_nloglik",
    "val_aft_nloglik",
    "val_c_index"
  )
  if (!all(required_columns %in% names(trajectory))) {
    stop("Training trajectory source is missing required columns.")
  }
  if (any(!is.finite(as.matrix(trajectory[, c(
    "training_progress_pct",
    "train_aft_nloglik",
    "val_aft_nloglik",
    "val_c_index"
  )])))) {
    stop("Training trajectory source contains non-finite values.")
  }

  trajectory_long <- trajectory %>%
    pivot_longer(
      cols = c(train_aft_nloglik, val_aft_nloglik, val_c_index),
      names_to = "metric",
      values_to = "value"
    )
  trajectory_grid <- interpolate_trajectories(trajectory_long)
  trajectory_summary <- trajectory_grid %>%
    group_by(split, metric, training_progress_pct) %>%
    summarise(
      median_value = median(value),
      min_value = min(value),
      max_value = max(value),
      num_seeds = n(),
      .groups = "drop"
    )

  roc_data <- bind_rows(
    read_roc_report(file.path(input_dir, "split42_consensus_roc.json"), "split 42"),
    read_roc_report(file.path(input_dir, "split43_consensus_roc.json"), "split 43")
  ) %>%
    mutate(split = factor(split, levels = c("split 42", "split 43")))
  horizons <- sort(unique(roc_data$horizon))
  if (!identical(horizons, c(36, 60, 84))) {
    stop("Expected ROC horizons 36, 60, and 84.")
  }
  roc_auc_summary <- roc_data %>%
    distinct(split, horizon, auc, num_cases, num_controls, num_excluded_censored)

  write_source_csv(trajectory, file.path(source_dir, "training_trajectories_raw.csv"))
  write_source_csv(trajectory_summary, file.path(source_dir, "training_trajectories_summary.csv"))
  write_source_csv(roc_data, file.path(source_dir, "roc_curve_points.csv"))
  write_source_csv(roc_auc_summary, file.path(source_dir, "roc_auc_summary.csv"))

  panel_a <- make_training_panel(
    trajectory_grid,
    trajectory_summary,
    metric_name = "train_aft_nloglik",
    tag = "a",
    title = "训练 AFT nloglik",
    subtitle = "细线为单 seed，粗线为五 seed 中位数",
    y_label = "Train AFT nloglik",
    log_y = TRUE,
    y_limits = c(2.7, 15),
    y_breaks = c(3, 4, 6, 10, 14)
  )
  panel_b <- make_training_panel(
    trajectory_grid,
    trajectory_summary,
    metric_name = "val_aft_nloglik",
    tag = "b",
    title = "验证 AFT nloglik",
    subtitle = "保存模型止于 validation 选定最优迭代",
    y_label = "Validation AFT nloglik",
    log_y = TRUE,
    y_limits = c(2.7, 15),
    y_breaks = c(3, 4, 6, 10, 14)
  )
  panel_c <- make_training_panel(
    trajectory_grid,
    trajectory_summary,
    metric_name = "val_c_index",
    tag = "c",
    title = "验证 C-index",
    subtitle = "两个 split 的五 seed 轨迹",
    y_label = "Validation C-index",
    log_y = FALSE,
    y_limits = c(0.64, 0.75),
    y_breaks = c(0.65, 0.70, 0.75)
  )
  panel_d <- make_roc_panel(roc_data, horizons[[1]], "d")
  panel_e <- make_roc_panel(roc_data, horizons[[2]], "e")
  panel_f <- make_roc_panel(roc_data, horizons[[3]], "f")
  panel_a <- panel_a + theme(legend.position = "top")
  panel_b <- panel_b + theme(legend.position = "none")
  panel_c <- panel_c + theme(legend.position = "none")
  panel_d <- panel_d + theme(legend.position = "none")
  panel_e <- panel_e + theme(legend.position = "none")
  panel_f <- panel_f + theme(legend.position = "none")

  mean_auc <- roc_auc_summary %>%
    group_by(split) %>%
    summarise(mean_auc = mean(auc), .groups = "drop")
  mean_auc_42 <- mean_auc$mean_auc[mean_auc$split == "split 42"]
  mean_auc_43 <- mean_auc$mean_auc[mean_auc$split == "split 43"]

  if (isTRUE(args$compact)) {
    panel_a <- panel_a + labs(title = NULL, subtitle = NULL, x = NULL)
    panel_b <- panel_b + labs(title = NULL, subtitle = NULL, x = "训练进度（% 最优迭代）")
    panel_c <- panel_c + labs(title = NULL, subtitle = NULL, x = NULL)
    panel_d <- panel_d +
      labs(
        title = sprintf("36 | AUC %.3f / %.3f",
          roc_auc_summary$auc[roc_auc_summary$split == "split 42" & roc_auc_summary$horizon == 36],
          roc_auc_summary$auc[roc_auc_summary$split == "split 43" & roc_auc_summary$horizon == 36]
        ),
        subtitle = NULL,
        x = NULL,
        y = "真阳性率"
      )
    panel_e <- panel_e +
      labs(
        title = sprintf("60 | AUC %.3f / %.3f",
          roc_auc_summary$auc[roc_auc_summary$split == "split 42" & roc_auc_summary$horizon == 60],
          roc_auc_summary$auc[roc_auc_summary$split == "split 43" & roc_auc_summary$horizon == 60]
        ),
        subtitle = NULL,
        x = "假阳性率",
        y = NULL
      )
    panel_f <- panel_f +
      labs(
        title = sprintf("84 | AUC %.3f / %.3f",
          roc_auc_summary$auc[roc_auc_summary$split == "split 42" & roc_auc_summary$horizon == 84],
          roc_auc_summary$auc[roc_auc_summary$split == "split 43" & roc_auc_summary$horizon == 84]
        ),
        subtitle = NULL,
        x = NULL,
        y = NULL
      )
    figure <- ((panel_a | panel_b | panel_c) / (panel_d | panel_e | panel_f)) +
      plot_layout(guides = "collect", heights = c(0.88, 1.12)) +
      plot_annotation(
        theme = theme(
          text = element_text(family = CN_FONT, colour = NEUTRAL_DARK),
          plot.margin = margin(4, 5, 4, 5, unit = "pt"),
          legend.position = "top",
          legend.justification = "center"
        )
      )
    output_base <- file.path(output_dir, "fig_training_dynamics_roc_compact_v3")
    save_figure(figure, output_base, height_mm = 105)
  } else {
    figure <- ((panel_a | panel_b | panel_c) / (panel_d | panel_e | panel_f)) +
      plot_layout(guides = "collect", heights = c(0.92, 1.06)) +
      plot_annotation(
        title = "新时间-拓扑模型的训练动态与时间依赖 ROC",
        subtitle = sprintf(
          "十个已保存 AFT 专家模型的真实回放；最终共识在三个时间点的平均 AUC 为 %.3f / %.3f",
          mean_auc_42,
          mean_auc_43
        ),
        caption = paste0(
          "ROC 为 cumulative/dynamic IPCW；曲线分别来自 split 42 与 split 43 的 α = 0.63 最终共识测试预测。",
          "训练横轴按各模型 validation 选定最优迭代归一化；topology_v6 为 synthetic/noisy augmented 数据。"
        ),
        theme = theme(
          text = element_text(family = CN_FONT, colour = NEUTRAL_DARK),
          plot.title = element_text(size = 10.8, face = "bold", margin = margin(b = 2)),
          plot.subtitle = element_text(size = 6.5, colour = NEUTRAL_MID, margin = margin(b = 5)),
          plot.caption = element_text(
            size = 5.15,
            colour = NEUTRAL_MID,
            hjust = 0,
            margin = margin(t = 5)
          ),
          plot.margin = margin(6, 6, 5, 6, unit = "pt"),
          legend.position = "top",
          legend.justification = "center"
        )
      )
    output_base <- file.path(output_dir, "fig_training_dynamics_roc_v3")
    save_figure(figure, output_base)
  }

  qa_notes <- c(
    "Figure contract",
    "Core conclusion: Saved AFT experts converge consistently across five model seeds in two splits, and the final validation-only consensus retains strong time-dependent discrimination.",
    "Archetype: quantitative 2 x 3 grid.",
    "Visual backend: R only (ggplot2 + patchwork; svglite/cairo_pdf/ragg export).",
    "Training source: exact replay of ten saved selected-expert XGBoost-AFT models; no model was retrained.",
    "Training x-axis: percentage of each model's validation-selected best iteration because selected boosters have different lengths.",
    "ROC metric: cumulative/dynamic IPCW ROC at horizons 36, 60, and 84.",
    "ROC model: final cross-split consensus alpha = 0.63, selected without test labels.",
    "Replicates: split seeds 42 and 43; model seeds 7, 21, 42, 123, and 2026.",
    "Dataset limitation: topology_v6 is synthetic/noisy augmented and is not an external clinical validation cohort.",
    "Source data: figure_source_data_training_roc/*.csv."
  )
  qa_name <- if (isTRUE(args$compact)) {
    "fig_training_dynamics_roc_compact_v3_qa.txt"
  } else {
    "fig_training_dynamics_roc_v3_qa.txt"
  }
  if (isTRUE(args$compact)) {
    qa_notes <- c(
      qa_notes,
      "Compact variant: no overall title, narrative subtitle, panel descriptions, or footer; only panel tags, metric/horizon labels, legend, AUC values, and axes are retained."
    )
  }
  writeLines(qa_notes, file.path(output_dir, qa_name), useBytes = TRUE)

  message(sprintf("Saved training/ROC figure bundle to %s", normalizePath(output_dir, winslash = "/")))
}

main()
