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
SIGNAL_TEAL <- "#2F857C"
WHITE <- "#FFFFFF"
HORIZON <- 60

parse_args <- function(args) {
  values <- list(
    input_dir = "outputs/current_mainline_v2/temporal_independent_v3/figure_diagnostics",
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

theme_nature_cn <- function(base_size = 6.5) {
  theme_classic(base_size = base_size, base_family = CN_FONT) +
    theme(
      axis.line = element_line(linewidth = 0.34, colour = NEUTRAL_DARK),
      axis.ticks = element_line(linewidth = 0.34, colour = NEUTRAL_DARK),
      axis.ticks.length = grid::unit(1.7, "pt"),
      axis.title = element_text(size = base_size, colour = NEUTRAL_DARK),
      axis.text = element_text(size = base_size - 0.4, colour = NEUTRAL_DARK),
      plot.title = element_text(
        size = base_size + 0.8,
        face = "bold",
        colour = NEUTRAL_DARK,
        margin = margin(b = 1.4)
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
      legend.title = element_text(size = 5.6),
      legend.text = element_text(size = 5.3)
    )
}

save_figure <- function(plot, output_base, width_mm = 183, height_mm = 118) {
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

normalize_range <- function(values) {
  value_range <- range(values, na.rm = TRUE)
  span <- diff(value_range)
  if (!is.finite(span) || span < 1e-12) {
    return(rep(0.5, length(values)))
  }
  (values - value_range[[1]]) / span
}

make_heatmap_panel <- function(
    heat_data,
    split_name,
    tag,
    c_index,
    high_risk_n,
    threshold_rank,
    show_y,
    show_legend) {
  panel_data <- filter(heat_data, split == split_name)
  panel <- ggplot(panel_data, aes(x = rank, y = track, fill = value)) +
    geom_raster() +
    geom_vline(
      xintercept = threshold_rank + 0.5,
      linewidth = 0.35,
      linetype = "22",
      colour = NEUTRAL_MID
    ) +
    scale_fill_gradient(
      low = "#F5F7FA",
      high = SIGNAL_BLUE,
      limits = c(0, 1),
      na.value = "#E4E6E8",
      name = "归一化值"
    ) +
    scale_x_continuous(
      breaks = c(1, 360, 720),
      labels = c("1", "360", "720"),
      limits = c(0.5, 720.5),
      expand = c(0, 0)
    ) +
    labs(
      tag = tag,
      title = sprintf("%s 风险排序", split_name),
      subtitle = sprintf("测试 C-index %.4f；validation 阈值划分高风险 n=%d", c_index, high_risk_n),
      x = "测试样本按新模型风险降序",
      y = NULL
    ) +
    theme_nature_cn() +
    theme(
      axis.line.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.y = if (show_y) element_text(size = 5.8) else element_blank(),
      plot.title = element_text(margin = margin(l = if (show_y) 0 else 7, b = 1.4)),
      plot.subtitle = element_text(margin = margin(l = if (show_y) 0 else 7, b = 4)),
      legend.position = if (show_legend) "bottom" else "none",
      legend.key.width = grid::unit(18, "pt"),
      legend.key.height = grid::unit(4, "pt")
    )
  panel
}

make_matrix_panel <- function(matrix_data, summary_row, split_name, tag) {
  panel_data <- filter(matrix_data, split == split_name)
  ggplot(panel_data, aes(x = risk_group, y = outcome_group, fill = group_fraction)) +
    geom_tile(colour = WHITE, linewidth = 0.75) +
    geom_text(
      aes(label = cell_label, colour = text_colour),
      family = CN_FONT,
      size = 2.45,
      fontface = "bold",
      lineheight = 0.92,
      show.legend = FALSE
    ) +
    scale_fill_gradient(
      low = "#F5F8F7",
      high = SIGNAL_TEAL,
      limits = c(0, 1),
      guide = "none"
    ) +
    scale_colour_identity() +
    coord_fixed() +
    labs(
      tag = tag,
      title = sprintf("%s 早发事件分层", split_name),
      subtitle = sprintf(
        "高风险 %.1f%% vs 低风险 %.1f%%\nRR %.2f；排除早期删失 n=%d",
        100 * summary_row$high_early_rate,
        100 * summary_row$low_early_rate,
        summary_row$risk_ratio,
        summary_row$excluded_censored
      ),
      x = "validation 阈值定义的风险组",
      y = sprintf("%.0f 时间单位结局", HORIZON)
    ) +
    theme_nature_cn() +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text.x = element_text(size = 5.8),
      axis.text.y = element_text(size = 5.8),
      plot.subtitle = element_text(size = 5.75, lineheight = 0.92, margin = margin(b = 4))
    )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  input_dir <- args$input_dir
  output_dir <- args$output_dir
  source_dir <- file.path(output_dir, "figure_source_data_risk_stratification")
  dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

  prediction_paths <- file.path(
    input_dir,
    c("split42_consensus_test_predictions.csv", "split43_consensus_test_predictions.csv")
  )
  missing <- prediction_paths[!file.exists(prediction_paths)]
  if (length(missing) > 0) {
    stop(sprintf("Required prediction source does not exist: %s", paste(missing, collapse = ", ")))
  }
  predictions <- bind_rows(lapply(prediction_paths, function(path) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  })) %>%
    mutate(
      split = factor(sprintf("split %d", split_seed), levels = c("split 42", "split 43")),
      risk_group = factor(
        if_else(selected_risk >= validation_median_risk_threshold, "高风险", "低风险"),
        levels = c("低风险", "高风险")
      ),
      horizon_status = case_when(
        event == 1 & time <= HORIZON ~ "早发事件",
        time > HORIZON ~ "无早发事件",
        TRUE ~ "提前删失"
      ),
      horizon_value = case_when(
        horizon_status == "早发事件" ~ 1,
        horizon_status == "无早发事件" ~ 0,
        TRUE ~ NA_real_
      )
    )
  if (any(!is.finite(predictions$time)) ||
      any(!is.finite(predictions$event)) ||
      any(!is.finite(predictions$selected_risk)) ||
      any(!is.finite(predictions$validation_median_risk_threshold))) {
    stop("Consensus prediction source contains non-finite values.")
  }
  if (any(!predictions$event %in% c(0, 1))) {
    stop("Event indicator must contain only 0 and 1.")
  }

  consensus_path <- file.path(
    dirname(input_dir),
    "cross_split_consensus",
    "cross_split_consensus_summary.json"
  )
  if (!file.exists(consensus_path)) {
    stop(sprintf("Consensus summary does not exist: %s", consensus_path))
  }
  consensus <- fromJSON(consensus_path, simplifyVector = TRUE)
  c_index_rows <- as_tibble(consensus$splits) %>%
    transmute(
      split = factor(sprintf("split %d", split_seed), levels = c("split 42", "split 43")),
      test_c_index = as.numeric(test_selected_c_index)
    )

  ordered <- predictions %>%
    group_by(split) %>%
    arrange(desc(selected_risk), .by_group = TRUE) %>%
    mutate(
      rank = row_number(),
      risk_norm = normalize_range(selected_risk),
      shorter_time_norm = 1 - normalize_range(time)
    ) %>%
    ungroup()
  heat_data <- ordered %>%
    select(
      split,
      split_seed,
      sample_id,
      rank,
      risk_group,
      "预测风险" = risk_norm,
      "观察到事件" = event,
      "较短观察时间" = shorter_time_norm,
      "60 内早发事件" = horizon_value
    ) %>%
    pivot_longer(
      cols = c("预测风险", "观察到事件", "较短观察时间", "60 内早发事件"),
      names_to = "track",
      values_to = "value"
    ) %>%
    mutate(
      track = factor(
        track,
        levels = c("预测风险", "观察到事件", "较短观察时间", "60 内早发事件")
      )
    )

  eligible <- predictions %>% filter(horizon_status != "提前删失") %>%
    mutate(
      outcome_group = factor(
        horizon_status,
        levels = c("无早发事件", "早发事件")
      )
    )
  matrix_data <- eligible %>%
    count(split, risk_group, outcome_group, name = "n") %>%
    complete(split, risk_group, outcome_group, fill = list(n = 0)) %>%
    group_by(split, risk_group) %>%
    mutate(
      group_total = sum(n),
      group_fraction = if_else(group_total > 0, n / group_total, 0),
      cell_label = sprintf("n=%d\n%.1f%%", n, 100 * group_fraction),
      text_colour = if_else(group_fraction >= 0.58, WHITE, NEUTRAL_DARK)
    ) %>%
    ungroup()

  rate_rows <- eligible %>%
    group_by(split, risk_group) %>%
    summarise(
      eligible_n = n(),
      early_rate = mean(horizon_status == "早发事件"),
      .groups = "drop"
    ) %>%
    pivot_wider(
      names_from = risk_group,
      values_from = c(eligible_n, early_rate),
      names_glue = "{.value}_{risk_group}"
    )
  excluded_rows <- predictions %>%
    group_by(split) %>%
    summarise(
      excluded_censored = sum(horizon_status == "提前删失"),
      test_n = n(),
      high_risk_n = sum(risk_group == "高风险"),
      threshold_rank = sum(risk_group == "高风险"),
      validation_threshold = first(validation_median_risk_threshold),
      .groups = "drop"
    )
  risk_summary <- rate_rows %>%
    left_join(excluded_rows, by = "split") %>%
    transmute(
      split,
      test_n,
      high_risk_n,
      threshold_rank,
      validation_threshold,
      high_eligible_n = eligible_n_高风险,
      low_eligible_n = eligible_n_低风险,
      high_early_rate = early_rate_高风险,
      low_early_rate = early_rate_低风险,
      risk_ratio = high_early_rate / low_early_rate,
      excluded_censored
    ) %>%
    left_join(c_index_rows, by = "split")

  write_source_csv(predictions, file.path(source_dir, "consensus_test_predictions_with_groups.csv"))
  write_source_csv(heat_data, file.path(source_dir, "risk_ranked_heatmap_values.csv"))
  write_source_csv(matrix_data, file.path(source_dir, "risk_stratification_matrix.csv"))
  write_source_csv(risk_summary, file.path(source_dir, "risk_stratification_summary.csv"))

  summary42 <- filter(risk_summary, split == "split 42")
  summary43 <- filter(risk_summary, split == "split 43")
  panel_a <- make_heatmap_panel(
    heat_data,
    "split 42",
    "a",
    summary42$test_c_index,
    summary42$high_risk_n,
    summary42$threshold_rank,
    show_y = TRUE,
    show_legend = TRUE
  )
  panel_b <- make_heatmap_panel(
    heat_data,
    "split 43",
    "b",
    summary43$test_c_index,
    summary43$high_risk_n,
    summary43$threshold_rank,
    show_y = FALSE,
    show_legend = FALSE
  )
  panel_c <- make_matrix_panel(matrix_data, summary42, "split 42", "c")
  panel_d <- make_matrix_panel(matrix_data, summary43, "split 43", "d")

  both_enriched <- all(risk_summary$high_early_rate > risk_summary$low_early_rate)
  subtitle <- if (both_enriched) {
    "validation 阈值固定后，两个独立 test split 的高风险组均富集 60 时间单位内早发事件"
  } else {
    "validation 阈值固定后，展示两个独立 test split 的 60 时间单位结局分布"
  }
  figure <- ((panel_a | panel_b) / (panel_c | panel_d)) +
    plot_layout(heights = c(0.92, 1.08), widths = c(1, 1)) +
    plot_annotation(
      title = "新时间-拓扑模型的风险排序与结局富集",
      subtitle = subtitle,
      caption = paste0(
        "风险阈值为各 split validation 风险中位数，固定后应用于 test；二分类矩阵排除 60 时间单位前删失者，",
        "热图保留全部 720 个测试样本（灰色表示该时间点结局未知）。\n",
        "数据：topology_v6（synthetic/noisy augmented）；当前结果不是外部临床队列验证。"
      ),
      theme = theme(
        text = element_text(family = CN_FONT, colour = NEUTRAL_DARK),
        plot.title = element_text(size = 10.9, face = "bold", margin = margin(b = 2)),
        plot.subtitle = element_text(size = 6.6, colour = NEUTRAL_MID, margin = margin(b = 6)),
        plot.caption = element_text(
          size = 5.2,
          colour = NEUTRAL_MID,
          hjust = 0,
          margin = margin(t = 5)
        ),
        plot.margin = margin(6, 6, 5, 6, unit = "pt")
      )
    )

  output_base <- file.path(output_dir, "fig_risk_stratification_replicated_v3")
  save_figure(figure, output_base)

  qa_notes <- c(
    "Figure contract",
    "Core conclusion: A validation-defined risk threshold separates early-event enrichment in both independent test splits.",
    "Archetype: paired risk-ranked heatmaps plus paired 2 x 2 stratification matrices.",
    "Visual backend: R only (ggplot2 + patchwork; svglite/cairo_pdf/ragg export).",
    "Risk score: final cross-split consensus alpha = 0.63.",
    "Threshold: median selected risk in the corresponding validation split; no test labels or test quantiles are used.",
    "Outcome definition: observed event by time 60 versus observed event-free beyond time 60.",
    "Censoring: samples censored at or before time 60 are excluded from matrices and shown as grey/unknown in heatmaps.",
    "Matrix percentages: within-risk-group percentages; each matrix column sums to 100%.",
    "Dataset limitation: topology_v6 is synthetic/noisy augmented and is not an external clinical validation cohort.",
    "Source data: figure_source_data_risk_stratification/*.csv."
  )
  writeLines(
    qa_notes,
    file.path(output_dir, "fig_risk_stratification_replicated_v3_qa.txt"),
    useBytes = TRUE
  )

  message(sprintf("Saved risk-stratification figure bundle to %s", normalizePath(output_dir, winslash = "/")))
  message(sprintf(
    "Early-event rates: split42 high %.3f vs low %.3f; split43 high %.3f vs low %.3f",
    summary42$high_early_rate,
    summary42$low_early_rate,
    summary43$high_early_rate,
    summary43$low_early_rate
  ))
}

main()
